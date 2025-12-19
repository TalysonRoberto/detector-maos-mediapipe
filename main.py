import cv2
import mediapipe as mp
import math

# ================= FUNÇÕES AUXILIARES =================

def distancia(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def angulo(a, b, c):
    # Calcula o ângulo ABC
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])

    if mag_ba * mag_bc == 0:
        return 0

    cos_angle = dot / (mag_ba * mag_bc)
    cos_angle = max(-1, min(1, cos_angle))

    return math.degrees(math.acos(cos_angle))

# ================= CONTADOR DE DEDOS =================

def contar_dedos(hand_landmarks):
    dedos = 0
    palma = hand_landmarks.landmark[0]

    # ---------- POLEGAR (ângulo + distância) ----------
    polegar_tip = hand_landmarks.landmark[4]
    polegar_ip  = hand_landmarks.landmark[3]
    polegar_mcp = hand_landmarks.landmark[2]

    dist_tip = distancia(polegar_tip, palma)
    dist_ip  = distancia(polegar_ip, palma)
    ang = angulo(polegar_tip, polegar_ip, polegar_mcp)

    # polegar só conta se estiver realmente estendido
    if dist_tip > dist_ip and ang > 160:
        dedos += 1

    # ---------- OUTROS DEDOS ----------
    dedos_info = [
        (8, 6),    # indicador
        (12, 10),  # médio
        (16, 14),  # anelar
        (20, 18)   # mindinho
    ]

    for tip, pip in dedos_info:
        if distancia(hand_landmarks.landmark[tip], palma) > distancia(hand_landmarks.landmark[pip], palma):
            dedos += 1

    return dedos

# ================= MAIN =================

def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    total = contar_dedos(hand_landmarks)

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    cv2.putText(
                        frame,
                        f"Dedos: {total}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3
                    )

            cv2.imshow("Detector de Maos", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

# ================= START =================

if __name__ == "__main__":
    main()
