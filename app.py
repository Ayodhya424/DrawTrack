import cv2
import numpy as np

# Try the current MediaPipe import path first, then fall back to the older module location.
try:
    from mediapipe.python.solutions import hands, drawing_utils
except ModuleNotFoundError:
    try:
        from mediapipe.solutions import hands, drawing_utils
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "MediaPipe Hands could not be imported. "
            "Install the compatible MediaPipe version in the active virtual environment with:\n"
            "    python -m pip install mediapipe==0.10.21\n"
            "Then rerun the app."
        ) from error

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Unable to open webcam. Please check your camera connection.")

# Create a white canvas (board)
board = np.ones((500, 640, 3), dtype=np.uint8) * 255
mask = np.zeros((500, 640), dtype=np.uint8)

# Drawing helpers

def can_place_shape(temp_mask):
    return cv2.countNonZero(cv2.bitwise_and(mask, temp_mask)) == 0


def add_shape_mask(temp_mask):
    global mask
    mask = cv2.bitwise_or(mask, temp_mask)

# Drawing configuration
colors = {
    'k': (0, 0, 0),
    'r': (0, 0, 255),
    'g': (0, 255, 0),
    'b': (255, 0, 0),
    'y': (0, 255, 255),
    'm': (255, 0, 255),
    'c': (255, 255, 0),
}
color_names = {
    'k': 'Black',
    'r': 'Red',
    'g': 'Green',
    'b': 'Blue',
    'y': 'Yellow',
    'm': 'Magenta',
    'c': 'Cyan',
}
current_color_key = 'k'
current_color = colors[current_color_key]
current_mode = 'write'
current_input = 'hand'
current_shape = 'circle'
shape_size = 30
thickness = 5
prev_point = None
shape_position = None
min_write_distance = 3


def count_extended_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    count = 0
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            count += 1
    return count


def find_pen_tip(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])
    blue_lower = np.array([90, 80, 70])
    blue_upper = np.array([140, 255, 255])

    red_mask = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    pen_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, red_mask2), blue_mask)

    kernel = np.ones((5, 5), np.uint8)
    pen_mask = cv2.morphologyEx(pen_mask, cv2.MORPH_OPEN, kernel)
    pen_mask = cv2.morphologyEx(pen_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(pen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 300:
            x, y, w, h = cv2.boundingRect(largest)
            return (x + w // 2, y + h // 2), pen_mask
    return None, pen_mask

# Initialize MediaPipe Hands
hands_detector = hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    hand_visible = False
    finger_count = 0
    draw_blocked_message = ''
    pen_tip = None
    pen_mask = None

    if current_input == 'pen':
        pen_tip, pen_mask = find_pen_tip(frame)
        if pen_tip is not None:
            shape_position = pen_tip
            hand_visible = True
            if current_mode == 'write':
                if prev_point is not None:
                    dx = pen_tip[0] - prev_point[0]
                    dy = pen_tip[1] - prev_point[1]
                    if np.hypot(dx, dy) >= min_write_distance:
                        cv2.line(board, prev_point, pen_tip, current_color, thickness)
                prev_point = pen_tip
            else:
                prev_point = None
            cv2.circle(frame, pen_tip, 8, current_color, 2)
        else:
            prev_point = None
            shape_position = None
    else:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                fingertip = hand_landmarks.landmark[8]  # index finger tip
                x, y = int(fingertip.x * w), int(fingertip.y * h)
                hand_visible = True
                shape_position = (x, y)
                finger_count = count_extended_fingers(hand_landmarks)

                if current_mode == 'write':
                    if finger_count >= 2:
                        prev_point = None
                    else:
                        if prev_point is not None:
                            dx = x - prev_point[0]
                            dy = y - prev_point[1]
                            if np.hypot(dx, dy) >= min_write_distance:
                                cv2.line(board, prev_point, (x, y), current_color, thickness)
                        prev_point = (x, y)
                else:
                    prev_point = None

                drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    hands.HAND_CONNECTIONS,
                )
                cv2.circle(frame, (x, y), 8, current_color, 2)
        else:
            prev_point = None
            shape_position = None

    if current_mode == 'shape' and shape_position is not None:
        if current_shape == 'circle':
            cv2.circle(frame, shape_position, shape_size, current_color, 2)
        elif current_shape == 'rectangle':
            cv2.rectangle(
                frame,
                (shape_position[0] - shape_size, shape_position[1] - shape_size),
                (shape_position[0] + shape_size, shape_position[1] + shape_size),
                current_color,
                2,
            )
        elif current_shape == 'triangle':
            points = np.array([
                [shape_position[0], shape_position[1] - shape_size],
                [shape_position[0] - shape_size, shape_position[1] + shape_size],
                [shape_position[0] + shape_size, shape_position[1] + shape_size],
            ])
            cv2.drawContours(frame, [points], 0, current_color, 2)
        elif current_shape == 'dot':
            cv2.circle(frame, shape_position, thickness, current_color, 2)

    info = [
        f"Mode: {current_mode} | Input: {current_input} | Color: {color_names[current_color_key]}",
        "Modes: w=write s=shape",
        "Input: h=hand p=pen",
        "Shapes: 2=circle 3=rect 4=tri 5=dot",
        "Press SPACE to place shape in shape mode",
        "Write mode allows overlap; shape mode blocks overlap",
        "2 fingers = stop writing (hand mode)",
        "Colors: r g b y m c k | +/- size",
        "c=clear q/ESC=quit",
    ]
    for i, text in enumerate(info):
        cv2.putText(frame, text, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if current_mode == 'write' and current_input == 'hand' and finger_count >= 2:
        cv2.putText(frame, "Two fingers detected: writing paused", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if current_input == 'pen' and pen_tip is None:
        cv2.putText(frame, "Pen not found - show a red or blue pen tip", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if draw_blocked_message:
        cv2.putText(frame, draw_blocked_message, (10, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Webcam", frame)
    cv2.imshow("Virtual Board", board)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        current_mode = 'write'
    elif key == ord('s'):
        current_mode = 'shape'
    elif key == ord('h'):
        current_input = 'hand'
    elif key == ord('p'):
        current_input = 'pen'
    elif key == ord('2'):
        current_shape = 'circle'
    elif key == ord('3'):
        current_shape = 'rectangle'
    elif key == ord('4'):
        current_shape = 'triangle'
    elif key == ord('5'):
        current_shape = 'dot'
    elif key == ord(' '):
        if current_mode == 'shape' and shape_position is not None:
            temp_mask = np.zeros_like(mask)
            if current_shape == 'circle':
                cv2.circle(temp_mask, shape_position, shape_size, 255, -1)
            elif current_shape == 'rectangle':
                cv2.rectangle(
                    temp_mask,
                    (shape_position[0] - shape_size, shape_position[1] - shape_size),
                    (shape_position[0] + shape_size, shape_position[1] + shape_size),
                    255,
                    -1,
                )
            elif current_shape == 'triangle':
                points = np.array([
                    [shape_position[0], shape_position[1] - shape_size],
                    [shape_position[0] - shape_size, shape_position[1] + shape_size],
                    [shape_position[0] + shape_size, shape_position[1] + shape_size],
                ])
                cv2.drawContours(temp_mask, [points], 0, 255, -1)
            elif current_shape == 'dot':
                cv2.circle(temp_mask, shape_position, thickness, 255, -1)

            if can_place_shape(temp_mask):
                if current_shape == 'circle':
                    cv2.circle(board, shape_position, shape_size, current_color, -1)
                elif current_shape == 'rectangle':
                    cv2.rectangle(
                        board,
                        (shape_position[0] - shape_size, shape_position[1] - shape_size),
                        (shape_position[0] + shape_size, shape_position[1] + shape_size),
                        current_color,
                        -1,
                    )
                elif current_shape == 'triangle':
                    points = np.array([
                        [shape_position[0], shape_position[1] - shape_size],
                        [shape_position[0] - shape_size, shape_position[1] + shape_size],
                        [shape_position[0] + shape_size, shape_position[1] + shape_size],
                    ])
                    cv2.drawContours(board, [points], 0, current_color, -1)
                elif current_shape == 'dot':
                    cv2.circle(board, shape_position, thickness, current_color, -1)
                add_shape_mask(temp_mask)
            else:
                draw_blocked_message = 'Overlap blocked in shape mode'
    elif key in map(ord, colors.keys()):
        current_color_key = chr(key)
        current_color = colors[current_color_key]
    elif key == ord('+') or key == ord('='):
        shape_size = min(shape_size + 5, 100)
        thickness = min(thickness + 1, 20)
    elif key == ord('-') or key == ord('_'):
        shape_size = max(shape_size - 5, 5)
        thickness = max(thickness - 1, 1)
    elif key == ord('c'):
        board[:] = 255
        mask[:] = 0
    elif key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
