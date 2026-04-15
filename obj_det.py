import cv2
import numpy as np
import torch
from ultralytics import FastSAM, YOLO

print(f"[Initialization] Loading YOLO-World and FastSAM globally... This may take a moment.")
device_id = 0 if torch.cuda.is_available() else -1
device_str = "cuda:0" if device_id == 0 else "cpu"
print(f"[Initialization] Using device: {device_str}")

# Load YOLOv8s-world model
detector = YOLO("yolov8s-world.pt")

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
            
        _global_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        _global_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        _global_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        _current_camera_index = camera_index
    return _global_cap

def locate_and_segment(target_class, camera_index=0):
    """
    Captures a frame from the camera, uses YOLO-World to find the target_class,
    and FastSAM to segment the object and find its centroid. Includes a live camera preview.
    """
    cap = get_camera(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # Set the target classes for YOLO-World
    detector.set_classes([target_class])
    
    print(f"Looking for '{target_class}' (Press 'q' in the window to abort)...")
    
    best_box = None
    final_frame = None
    final_annotated_frame = None
    frame_count = 0

    # Loop the camera feed until the object is found
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not grab a frame from the camera.")
            break

        frame_count += 1

        # 1. Detect bounding box with YOLO-World
        try:
            results = detector.predict(frame, conf=0.1, device=device_str, verbose=False)
        except Exception as e:
            print(f"Vision Inference Error: {e}")
            results = []
            
        annotated_frame = frame.copy()
        best_match = None
        best_conf = 0.0
        
        if results and len(results) > 0 and len(results[0].boxes) > 0:
            # Sort boxes by highest confidence
            boxes = results[0].boxes
            conf_list = boxes.conf.cpu().numpy()
            
            best_idx = np.argmax(conf_list)
            best_conf = conf_list[best_idx]
            
            # xmin, ymin, xmax, ymax
            best_box_tensor = boxes.xyxy[best_idx]
            best_box_tmp = best_box_tensor.cpu().numpy().tolist()
            best_match = True
            
            # Draw the bounding box for live view
            cv2.rectangle(annotated_frame, 
                          (int(best_box_tmp[0]), int(best_box_tmp[1])), 
                          (int(best_box_tmp[2]), int(best_box_tmp[3])), 
                          (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{target_class}: {best_conf:.2f}",
                        (int(best_box_tmp[0]), int(best_box_tmp[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Visualize the current frame
        cv2.imshow("Robot Vision - YOLO-World", annotated_frame)

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
            if best_conf > 0.15:
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
        
        cv2.imshow("Robot Vision - YOLO-World", final_annotated_frame)
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
