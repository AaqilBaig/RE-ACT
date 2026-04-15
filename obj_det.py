import cv2
import numpy as np
from ultralytics import FastSAM
from transformers import pipeline
from PIL import Image

print(f"[Initialization] Loading OWLv2 and FastSAM globally... This may take a moment.")
detector = pipeline(task="zero-shot-object-detection", model="google/owlv2-base-patch16")
fast_sam = FastSAM("FastSAM-s.pt")
print(f"[Initialization] Models loaded successfully.")

def locate_and_segment(target_class, camera_index=1):
    """
    Captures a frame from the camera, uses OWLv2 (a CLIP-based detector) to find the target_class,
    and FastSAM to segment the object and find its centroid. Includes a live camera preview.
    """
    # Capture frame (Removed cv2.CAP_DSHOW to fix Windows camera indexing)
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

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
        
        # Convert BGR OpenCV frame to RGB PIL image for transformers pipeline
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # 1. Detect bounding box with OWL-ViT
        try:
            results = detector(pil_img, candidate_labels=[target_class])
        except Exception as e:
            print(f"Vision Inference Error: {e}")
            results = []
            
        annotated_frame = frame.copy()
        best_match = None
        
        if results and len(results) > 0:
            # Sort by highest score
            best_match = max(results, key=lambda x: x['score'])
            best_conf = best_match['score']
            
            box = best_match['box']
            # Format expected by FastSAM: [xmin, ymin, xmax, ymax]
            best_box_tmp = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
            
            # Draw the bounding box for live view
            cv2.rectangle(annotated_frame, 
                          (int(box['xmin']), int(box['ymin'])), 
                          (int(box['xmax']), int(box['ymax'])), 
                          (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{target_class}: {best_conf:.2f}",
                        (int(box['xmin']), int(box['ymin']) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Visualize the current frame
        cv2.imshow("Robot Vision - CLIP Detector", annotated_frame)

        # Break if user presses 'q' manually
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[Vision Debug] User aborted via 'q' key.")
            break

        # Check if the target was actually found in this frame
        if best_match:
            if frame_count % 10 == 0:
                print(f"[Vision Debug] Frame {frame_count}: Model sees '{target_class}' with confidence ({best_conf:.2f}).")
            
            # Additional validation threshold for CLIP-based detection
            if best_conf > 0.1:
                print(f"[Vision Debug] Detection locked! Confidence: {best_conf:.2f}")
                # Lock in this frame for SAM
                final_frame = frame.copy()
                final_annotated_frame = annotated_frame.copy()
                best_box = best_box_tmp
                print(f"Found '{target_class}' bounding box: {best_box}")
                
                cv2.waitKey(500) # Briefly pause on the found frame
                break
            else:
                if frame_count % 10 == 0:
                    print(f"[Vision Debug] Phantom object detected but confidence ({best_conf:.2f}) is too low to trigger lock...")

    cap.release()

    if not best_box:
        print(f"'{target_class}' not found or aborted.")
        cv2.destroyAllWindows()
        return None

    # 2. Segment within the bounding box using FastSAM
    sam_results = fast_sam(final_frame, bboxes=[best_box], verbose=False)
    
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
        
        cv2.imshow("Robot Vision - CLIP Detector", final_annotated_frame)
        cv2.waitKey(1500) # Display the calculated grasp point for 1.5 seconds
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
