import cv2
import gradio as gr

cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2(200, 16)

if not cap.isOpened():
    print("Error opening video file")
else:
    while cap.isOpened():
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
        # Apply background subtraction
            fg_mask = backSub.apply(frame, learningRate=0.7)
        
            # contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # print(contours)
            # frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            # Display the resulting frame
            
            # print(contours)
            
            retval, mask_thresh = cv2.threshold(fg_mask, 120, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            # Apply erosion
            mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

            contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_contour_area = 500  # Define your minimum area threshold
            large_contours1 = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

            frame_1 = cv2.drawContours(frame, large_contours1, -1, (0, 255, 0), 2)
            # Display the resulting frame
            cv2.imshow('Mask', mask_eroded)
            cv2.imshow('Contour', frame_1)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

cap.release()
cv2.destroyAllWindows()