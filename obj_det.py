import cv2
import numpy as np
import torch
import threading
from ultralytics import FastSAM
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image

print(f"[Initialization] Loading OWL-v2 and FastSAM globally... This may take a moment.")
device_id = 0 if torch.cuda.is_available() else -1
device_str = "cuda:0" if device_id == 0 else "cpu"

if device_id == 0:
    torch.backends.cudnn.benchmark = True

# Enable CPU multithreading optimizations explicitly if no GPU is found
if device_str == "cpu":
    torch.set_num_threads(4) # Limit PyTorch to 4 threads to prevent it from chocking the OS and locking the CPU cache 
print(f"[Initialization] Using device: {device_str}")

# Load OWL-v2 model
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")

detector = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device_str)
detector.eval()

fast_sam = FastSAM("FastSAM-s.pt")
# NOTE: To enforce GPU for FastSAM on inference, use device=device_str in fast_sam() call later if needed.
print(f"[Initialization] Models loaded successfully.")

# Global camera instance to prevent startup latency on every grasp
_global_cap = None
_current_camera_index = None

def get_camera(camera_index):
    global _global_cap, _current_camera_index
    if _global_cap is None or _current_camera_index != camera_index:
        if _global_cap is not None:
            _global_cap.release()
        print(f"[Initialization] Opening camera {camera_index}...")
        # Use DirectShow (CAP_DSHOW) backend on Windows to bypass slow auto-probing
        _global_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not _global_cap.isOpened():
            # Fallback if DirectShow fails
            _global_cap = cv2.VideoCapture(camera_index)
            
        # Set MJPG codec to speed up camera initialization/loading significantly on Windows
        _global_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        _global_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        _global_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        _global_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        _global_cap.set(cv2.CAP_PROP_FPS, 30)
        _current_camera_index = camera_index
    return _global_cap

def locate_and_segment(target_class, camera_index=0):
    """
    Captures a frame from the camera, uses OWL-v2 to find the target_class,
    and FastSAM to segment the object and find its centroid. Includes a live camera preview.
    """
    cap = get_camera(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    print(f"Looking for '{target_class}' (Press 'q' in the window to abort)...")
    
    best_box = None
    final_frame = None
    final_annotated_frame = None
    frame_count = 0

    best_box_tmp = None
    best_conf = 0.0
    best_match = False
    stable_hits = 0
    # Track motion between inference updates so box follows moving objects smoothly.
    track_template = None
    track_box = None
    track_last_update = 0
    track_miss_count = 0

    target_lower = target_class.lower()
    small_object_keywords = ("key", "coin", "tape", "pen", "screw", "bolt", "nut")
    is_small_object = any(k in target_lower for k in small_object_keywords)
    detection_threshold = 0.18 if is_small_object else 0.14
    lock_threshold = 0.24 if is_small_object else 0.18
    min_stable_hits = 0 if is_small_object else 1

    def box_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = a_area + b_area - inter
        return (inter / union) if union > 0 else 0.0

    inference_running = [False]

    def run_inference(frame_copy, target):
        nonlocal best_box_tmp, best_conf, best_match, stable_hits
        nonlocal track_template, track_box, track_last_update, track_miss_count
        try:
            # Keep aspect ratio to avoid shape distortions that can cause false positives
            infer_w, infer_h = 256, 192
            small_frame = cv2.resize(frame_copy, (infer_w, infer_h))
            image = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
            texts = [[f"a photo of a {target}"]]
            inputs = processor(text=texts, images=image, return_tensors="pt").to(device_str)
            if device_id == 0:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            
            # Use inference_mode to accelerate the PyTorch model computationally 
            with torch.inference_mode():
                if device_id == 0:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = detector(**inputs)
                else:
                    outputs = detector(**inputs)
                
            target_sizes = torch.tensor([image.size[::-1]]).to(device_str)
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=detection_threshold)
            
            if results and len(results) > 0:
                scores = results[0]["scores"]
                boxes = results[0]["boxes"]
                
                if len(scores) > 0:
                    frame_h, frame_w = frame_copy.shape[:2]
                    max_area_ratio = 0.22 if is_small_object else 0.80
                    min_area_ratio = 0.0002

                    candidate_box = None
                    candidate_conf = 0.0
                    sorted_idxs = torch.argsort(scores, descending=True)

                    for idx_tensor in sorted_idxs:
                        idx = idx_tensor.item()
                        conf = scores[idx].item()
                        raw_box = boxes[idx].cpu().numpy().tolist()

                        scale_x = frame_w / infer_w
                        scale_y = frame_h / infer_h
                        box = [raw_box[0] * scale_x, raw_box[1] * scale_y, raw_box[2] * scale_x, raw_box[3] * scale_y]

                        bw = max(1.0, box[2] - box[0])
                        bh = max(1.0, box[3] - box[1])
                        area_ratio = (bw * bh) / float(frame_w * frame_h)

                        # Filter out implausibly large/small regions for the target object type.
                        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                            continue

                        candidate_box = box
                        candidate_conf = conf
                        break

                    if candidate_box is not None:
                        if best_box_tmp is not None and box_iou(best_box_tmp, candidate_box) > 0.30:
                            stable_hits = min(stable_hits + 1, 3)
                        else:
                            stable_hits = 0

                        best_box_tmp = candidate_box
                        best_conf = candidate_conf
                        best_match = True
                        track_box = candidate_box.copy()
                        cx1, cy1, cx2, cy2 = [int(v) for v in candidate_box]
                        cx1 = max(0, min(cx1, frame_w - 1))
                        cy1 = max(0, min(cy1, frame_h - 1))
                        cx2 = max(cx1 + 1, min(cx2, frame_w))
                        cy2 = max(cy1 + 1, min(cy2, frame_h))
                        gray_full = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                        roi = gray_full[cy1:cy2, cx1:cx2]
                        if roi.size > 0:
                            track_template = roi.copy()
                            track_last_update = frame_count
                            track_miss_count = 0
                    else:
                        if track_template is None:
                            best_match = False
        except Exception as e:
            print(f"Vision Inference Error: {e}")
        finally:
            inference_running[0] = False

    # Loop the camera feed until the object is found
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not grab a frame from the camera.")
            break

        frame_count += 1
        annotated_frame = frame.copy()

        # Run OWL-v2 instance asynchronously to completely avoid main-thread blocking and camera drops
        if not inference_running[0]:
            inference_running[0] = True
            threading.Thread(target=run_inference, args=(frame.copy(), target_class), daemon=True).start()

        # Fast motion tracking between OWL-v2 updates.
        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if best_match and best_box_tmp is not None:
            x1, y1, x2, y2 = [int(v) for v in best_box_tmp]
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(x1 + 1, min(x2, frame_w))
            y2 = max(y1 + 1, min(y2, frame_h))

            # Rebuild template when detection updates or tracker is stale.
            if track_box is None or box_iou(track_box, [x1, y1, x2, y2]) < 0.60:
                roi = gray[y1:y2, x1:x2]
                if roi.size > 0:
                    track_template = roi.copy()
                    track_box = [float(x1), float(y1), float(x2), float(y2)]
                    track_last_update = frame_count

        if track_template is not None and track_box is not None:
            tx1, ty1, tx2, ty2 = [int(v) for v in track_box]
            bw = max(1, tx2 - tx1)
            bh = max(1, ty2 - ty1)
            margin_x = max(12, int(bw * 0.75))
            margin_y = max(12, int(bh * 0.75))

            sx1 = max(0, tx1 - margin_x)
            sy1 = max(0, ty1 - margin_y)
            sx2 = min(frame_w, tx2 + margin_x)
            sy2 = min(frame_h, ty2 + margin_y)

            search = gray[sy1:sy2, sx1:sx2]
            if search.shape[0] >= track_template.shape[0] and search.shape[1] >= track_template.shape[1]:
                corr = cv2.matchTemplate(search, track_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(corr)

                # Accept only reasonable correlation to avoid drift.
                if max_val >= 0.45:
                    nx1 = sx1 + max_loc[0]
                    ny1 = sy1 + max_loc[1]
                    nx2 = nx1 + track_template.shape[1]
                    ny2 = ny1 + track_template.shape[0]
                    track_box = [float(nx1), float(ny1), float(nx2), float(ny2)]
                    best_box_tmp = track_box.copy()
                    track_last_update = frame_count
                    best_match = True
                    track_miss_count = 0

                    # Refresh template occasionally for robustness against lighting/pose change.
                    if frame_count % 8 == 0:
                        new_roi = gray[ny1:ny2, nx1:nx2]
                        if new_roi.shape == track_template.shape and new_roi.size > 0:
                            track_template = new_roi.copy()
                else:
                    track_miss_count += 1
                    if track_miss_count > 10:
                        track_template = None
                        track_box = None
                        best_match = False

        # Draw the bounding box for live view using the most recently known detection
        if best_match and best_box_tmp is not None:
            cv2.rectangle(annotated_frame, 
                          (int(best_box_tmp[0]), int(best_box_tmp[1])), 
                          (int(best_box_tmp[2]), int(best_box_tmp[3])), 
                          (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{target_class}: {best_conf:.2f}",
                        (int(best_box_tmp[0]), int(best_box_tmp[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Visualize the current frame
        cv2.imshow("Robot Vision - OWL-v2", annotated_frame)

        # Break if user presses 'q' manually
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[Vision Debug] User aborted via 'q' key.")
            break

        # Check if the target was actually found in this frame
        if best_match:
            if frame_count % 10 == 0:
                print(f"[Vision Debug] Frame {frame_count}: Model sees '{target_class}' with confidence ({best_conf:.2f}).")
            
            # Lock in if confidence exceeds threshold (can increase this for YOLO)
            if best_conf > lock_threshold and stable_hits >= min_stable_hits:
                print(f"[Vision Debug] Detection locked! Confidence: {best_conf:.2f}")
                # Lock in this frame for SAM
                final_frame = frame.copy()
                final_annotated_frame = annotated_frame.copy()
                best_box = best_box_tmp
                print(f"Found '{target_class}' bounding box: {best_box}")
                
                cv2.waitKey(1) # Briefly pump the event loop
                break
            else:
                if frame_count % 10 == 0:
                    print(f"[Vision Debug] Phantom object detected but confidence ({best_conf:.2f}) is too low to trigger lock...")

    # We do NOT release the camera here so it stays open for the next call.
    # cap.release() 

    if not best_box:
        print(f"'{target_class}' not found or aborted.")
        cv2.destroyAllWindows()
        return None

    # 2. Segment within the bounding box using FastSAM
    sam_results = fast_sam(final_frame, bboxes=[best_box], verbose=False, device=device_str)
    
    if not sam_results or sam_results[0].masks is None:
        print("Failed to generate a segmentation mask.")
        cv2.destroyAllWindows()
        return None

    # Get the mask (binary array)
    mask = sam_results[0].masks.data[0].cpu().numpy()
    
    # FastSAM outputs masks at network resolution, resize to original frame size
    if mask.shape != (final_frame.shape[0], final_frame.shape[1]):
        mask = cv2.resize(mask, (final_frame.shape[1], final_frame.shape[0]))
    mask = (mask > 0).astype(np.uint8)

    # 3. Calculate the centroid using image moments
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        print(f"Calculated centroid for '{target_class}': ({cX}, {cY})")
        
        # Draw the grasp centroid point visually on the final frame
        cv2.circle(final_annotated_frame, (cX, cY), 7, (0, 0, 255), -1) # Red dot
        cv2.putText(final_annotated_frame, f"Grasp: ({cX}, {cY})", (cX - 40, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Robot Vision - OWL-v2", final_annotated_frame)
        cv2.waitKey(1) # Display the calculated grasp point
        cv2.destroyAllWindows()
        
        return (cX, cY)
    else:
        print("Error calculating centroid (zero area mask).")
        cv2.destroyAllWindows()
        return None

if __name__ == "__main__":
    # Test the function individually
    print("Testing obj_det.py individually...")
    coords = locate_and_segment("person")
    print("Coordinates:", coords)
