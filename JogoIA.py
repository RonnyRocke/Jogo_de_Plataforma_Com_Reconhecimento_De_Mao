import cv2
import mediapipe as mp
import numpy as np
import random

# Inicializa o MediaPipe Hands e a captura de vídeo
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Configura o jogo
win_width, win_height = 800, 400
player_width, player_height = 50, 50
player_x, player_y = win_width // 4, win_height - player_height - 50
player_vel = 10  # Velocidade do jogador
player_jump = False
jump_count = 10
obstacle_width, obstacle_height = 50, 50
ground_y = win_height - player_height  # Posição do solo
obstacles = [(win_width - 300, ground_y)]  # Distância aumentada entre os obstáculos
score = 0
gravity = 0.8  # Gravidade ajustada para uma queda mais natural

# Partículas
particles = []

# Função para detectar se a mão está fechada
def is_hand_closed(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    
    # Verifica se todos os dedos estão abaixados (indicando que a mão está fechada)
    return (thumb_tip.y > landmarks[mp_hands.HandLandmark.THUMB_IP].y and
            index_finger_tip.y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
            middle_finger_tip.y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            ring_finger_tip.y > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y and
            pinky_tip.y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y)

# Função para desenhar o jogador e obstáculos
def draw_game(img, score, camera_frame):
    global player_y
    # Desenha o fundo
    img[:] = (255, 255, 255)  # Fundo branco

    # Desenha o jogador
    cv2.rectangle(img, (player_x, int(player_y)), (player_x + player_width, int(player_y) + player_height), (0, 0, 0), -1)

    # Desenha os obstáculos
    for (obstacle_x, obstacle_y) in obstacles:
        cv2.rectangle(img, (obstacle_x, obstacle_y), (obstacle_x + obstacle_width, obstacle_y + obstacle_height), (0, 0, 255), -1)

    # Desenha as partículas
    for particle in particles:
        cv2.circle(img, (int(particle[0]), int(particle[1])), 5, particle[2], -1)

    # Desenha a pontuação
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f'Score: {score}', (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Adiciona a imagem da câmera no canto inferior direito
    camera_frame = cv2.resize(camera_frame, (200, 150))  # Reduz o tamanho da imagem da câmera
    img[win_height - 150:, win_width - 200:] = camera_frame

def draw_menu(img):
    # Desenha o fundo
    img[:] = (255, 255, 255)  # Fundo branco
    
    # Desenha o título
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Autor: Ronny Rocke', (50, 100), font, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Desenha as opções do menu
    cv2.putText(img, 'Pressione "I" para Iniciar', (50, 200), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, 'Pressione "Q" para Sair', (50, 250), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

def update_particles():
    global particles
    # Atualiza as partículas
    new_particles = []
    for particle in particles:
        x, y, color, x_vel, y_vel, lifetime = particle
        x += x_vel
        y += y_vel
        lifetime -= 1
        if lifetime > 0:
            new_particles.append((x, y, color, x_vel, y_vel, lifetime))
    particles = new_particles

def create_explosion(x, y):
    global particles
    particles = []
    for _ in range(100):  # Cria 100 partículas para a explosão
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(5, 15)
        x_vel = speed * np.cos(angle)
        y_vel = speed * np.sin(angle)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        particles.append((x, y, color, x_vel, y_vel, random.randint(20, 50)))

def game_loop():
    global player_x, player_y, player_jump, jump_count, obstacles, score, ground_y, particles

    cap = cv2.VideoCapture(0)
    previous_hand_state = False  # Estado da mão na iteração anterior
    game_active = False  # Controla se o jogo está ativo ou não

    while True:
        success, img = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        game_img = np.zeros((win_height, win_width, 3), dtype=np.uint8)

        if not game_active:
            draw_menu(game_img)
        else:
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    # Desenha as mãos para depuração
                    mp_draw.draw_landmarks(img, landmarks)

                    hand_closed = is_hand_closed(landmarks.landmark)

                    # Verifica se a mão foi fechada e não estava fechada na iteração anterior
                    if hand_closed and not previous_hand_state:
                        player_jump = True

                    previous_hand_state = hand_closed

            # Atualiza a posição do jogador
            if player_jump:
                if jump_count >= -10:
                    neg = 1 if jump_count < 0 else -1
                    player_y -= (jump_count ** 2) * 0.4 * neg  # Ajusta a força do pulo
                    jump_count -= 1
                    if player_y <= ground_y - 100:  # Limita o máximo de subida
                        player_y = ground_y - 100
                else:
                    player_jump = False
                    jump_count = 10

            if not player_jump and player_y < ground_y:
                player_y += gravity  # Gravidade simulada

            # Verifica colisão com obstáculos
            collision = False
            for (obstacle_x, obstacle_y) in obstacles:
                if (player_x + player_width > obstacle_x and
                    player_x < obstacle_x + obstacle_width and
                    player_y + player_height > obstacle_y and
                    player_y < obstacle_y + obstacle_height):
                    # Se o jogador colidir com um obstáculo, cria partículas e reinicia o jogo
                    create_explosion(player_x + player_width // 2, player_y + player_height // 2)
                    player_x, player_y = win_width // 4, ground_y
                    obstacles = [(win_width - 300, ground_y)]  # Reinicia os obstáculos
                    score = 0
                    collision = True
                    break

            # Atualiza as partículas
            update_particles()

            # Move obstáculos para a esquerda
            if not collision:
                obstacles = [(x - player_vel, y) for (x, y) in obstacles]

                # Adiciona um novo obstáculo se o anterior saiu da tela
                if len(obstacles) == 0 or obstacles[-1][0] < win_width - 400:
                    obstacles.append((win_width, ground_y))
                    score += 1  # Incrementa a pontuação

            # Desenha o jogador e os obstáculos
            draw_game(game_img, score, img)

        # Exibe o resultado
        cv2.imshow("Simple Game", game_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('i'):  # Pressione 'I' para iniciar o jogo
            game_active = True
        elif key == ord('q'):  # Pressione 'Q' para sair
            break

    cap.release()
    cv2.destroyAllWindows()

# Inicia o jogo
game_loop()
