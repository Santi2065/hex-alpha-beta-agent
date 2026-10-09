import numpy as np
import heapq # una cola de prioridad para A*
import time

class GrobaAgent:
    def __init__(self):
        """
        Inicializa el agente con las estructuras necesarias:
        - Tabla de transposición para evitar recálculos
        - Caché de caminos para optimizar A*
        - Direcciones para explorar conexiones en el tablero hexagonal
        - Tabla Zobrist para hashing eficiente
        """
        self.board_size = None
        self.max_depth = 7
        self.transposition_table = {}
        self.path_cache = {}  # Caché específico para cálculos de caminos
        self.directions = [(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1)]
        self.zobrist_table = None

    def initialize_zobrist(self, size):
        """Inicializa tabla Zobrist para el tamaño de tablero actual"""
        self.zobrist_table = {}
        for r in range(size):
            for c in range(size):
                for p in [1, -1]:  # Los dos jugadores
                    self.zobrist_table[(r, c, p)] = np.random.randint(0, 2**64)

    def get_zobrist_hash(self, board):
        """Calcula hash Zobrist para el tablero"""
        if self.zobrist_table is None:
            self.initialize_zobrist(self.board_size)
        
        h = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] != 0:
                    h ^= self.zobrist_table[(r, c, board[r, c])]
        return h

    def action(self, board):
        """Punto de entrada del agente que garantiza respuesta dentro del límite de tiempo"""
        start_time = time.time()
        time_limit = 25.0 
        
        if self.board_size is None:
            self.board_size = len(board)
        
        # Verificar movimiento ganador con control de tiempo
        current_time = time.time() - start_time
        if current_time < time_limit * 0.5:  # Solo intentar si hay tiempo suficiente
            winning_move = self.check_potential_win(board, 1, start_time, time_limit)
            if winning_move is not None:
                return winning_move
        
        # Verificar bloqueo con control de tiempo
        current_time = time.time() - start_time
        if current_time < time_limit * 0.7:  # Solo intentar si hay tiempo suficiente
            blocking_move = self.check_potential_win(board, -1, start_time, time_limit)
            if blocking_move is not None:
                return blocking_move
        
        # Verificar tiempo restante para búsqueda
        current_time = time.time() - start_time
        remaining_time = time_limit - current_time
        
        # Si queda poco tiempo, usar una respuesta rápida
        if remaining_time < 5.0:
            return self.get_fast_move(board)
        
        # Continuar con búsqueda normal pasando el tiempo inicial
        return self.find_best_move_with_time(board, start_time, remaining_time)
        
    def name(self):
        return {"nombre": "Santiago", "apellido": "Groba", "legajo": 35514}

    def __str__(self):
        return "Theseus"

    def reset(self):
        self.transposition_table = {}
        self.path_cache = {}

    def get_empty_cells(self, board):
        return np.argwhere(board == 0)
    
    def make_move(self, board, move, player):
        new_board = board.copy()
        row, col = divmod(move, self.board_size)
        new_board[row][col] = player
        return new_board

    # Evaluación basada en distancia de caminos: favorece posiciones donde
    # nuestro camino es corto y el del oponente es largo
    def evaluate(self, board):
        """Versión optimizada de la evaluación"""
        # Verificar victoria/derrota inmediata
        player_win = self.has_connecting_path(board, 1)
        if player_win:
            return 1000
        
        opponent_win = self.has_connecting_path(board, -1)
        if opponent_win:
            return -1000
        
        # Solo si no hay victoria/derrota inmediata, calcular distancias
        player_distance = self.shortest_path_length(board, 1)
        opponent_distance = self.shortest_path_length(board, -1)
        
        return opponent_distance - player_distance

    def has_connecting_path(self, board, player):
        """Verificación rápida de victoria usando los bordes"""
        # Definir bordes según jugador
        if player == 1:  # Horizontal (izq/der)
            left_border = [(i, 0) for i in range(self.board_size)]
            right_border = [(i, self.board_size-1) for i in range(self.board_size)]
            
            # Verificar conexiones desde borde izquierdo
            for start in left_border:
                r, c = start
                if board[r, c] == player:
                    # Buscar camino a cualquier punto del borde derecho
                    for end in right_border:
                        if self.has_path(board, player, start, end):
                            return True
        else:  # Vertical (arr/abj)
            top_border = [(0, i) for i in range(self.board_size)]
            bottom_border = [(self.board_size-1, i) for i in range(self.board_size)]
            
            for start in top_border:
                r, c = start
                if board[r, c] == player:
                    for end in bottom_border:
                        if self.has_path(board, player, start, end):
                            return True
        
        return False

    def shortest_path_length(self, board, player):
        # Crear clave para caché
        board_hash = hash(board.tobytes())
        cache_key = (board_hash, player)
        
        # Verificar caché
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Tablero de costos: 0 para celdas propias, 1 para vacías, infinito para oponente
        cost_board = np.ones((self.board_size, self.board_size)) * float('inf')
        cost_board[board == player] = 0
        cost_board[board == 0] = 1
        
        # Definir bordes según jugador (horizontal o vertical)
        if player == 1:  # Horizontal
            start_edge = [(i, 0) for i in range(self.board_size)]
            end_edge = [(i, self.board_size-1) for i in range(self.board_size)]
        else:  # Vertical
            start_edge = [(0, i) for i in range(self.board_size)]
            end_edge = [(self.board_size-1, i) for i in range(self.board_size)]
        
        # Verificar si ya existe un camino completo
        for start in start_edge:
            for end in end_edge:
                if self.has_path(board, player, start, end):
                    return 0
        
        # Almacenar en caché antes de retornar
        result = self.astar_shortest_path(cost_board, start_edge, end_edge)
        self.path_cache[cache_key] = result
        return result

    # Verifica si existe un camino continuo de piezas del jugador usando DFS
    def has_path(self, board, player, start, end):
        visited = set()
        
        def dfs(position):
            if position in visited:
                return False
            
            row, col = position
            if not (0 <= row < self.board_size and 0 <= col < self.board_size):
                return False
                
            if board[row][col] != player:
                return False
                
            if position == end:
                return True
                
            visited.add(position)
            
            for dr, dc in self.directions:
                if dfs((row + dr, col + dc)):
                    return True
                    
            return False
        
        return dfs(start)

    def astar_shortest_path(self, cost_board, start_positions, end_positions):
        """
        Implementa A* para encontrar el camino más corto entre bordes.
        
        La heurística valora:
        - Distancia directa al borde opuesto
        - Presencia de fichas propias que pueden facilitar el camino
        
        Retorna la longitud del camino más corto o un valor grande si no existe.
        """
        end_set = set(tuple(pos) for pos in end_positions)
        
        def heuristic(pos, is_horizontal, board):
            row, col = pos
            if is_horizontal:
                base_distance = self.board_size - col
            else:
                base_distance = self.board_size - row
            
            player = 1 if is_horizontal else -1
            density_bonus = 0
            
            # Contar piezas propias en dirección al objetivo
            if is_horizontal:
                for c in range(col+1, self.board_size):
                    if board[row, c] == player:
                        density_bonus += 1
            else:
                for r in range(row+1, self.board_size):
                    if board[r, col] == player:
                        density_bonus += 1
            
            return max(1, base_distance - density_bonus * 0.5)
        
        is_horizontal = start_positions[0][1] == 0
        g_score = np.ones((self.board_size, self.board_size)) * float('inf')
        open_set = []
        
        for start in start_positions:
            row, col = start
            if cost_board[row, col] < float('inf'):
                g_score[row, col] = cost_board[row, col]
                f_score = g_score[row, col] + heuristic(start, is_horizontal, cost_board)
                heapq.heappush(open_set, (f_score, g_score[row, col], start))
        
        while open_set:
            _, current_g, (row, col) = heapq.heappop(open_set)
            
            if (row, col) in end_set:
                return current_g
                
            if current_g > g_score[row, col]:
                continue
                
            for dr, dc in self.directions:
                new_row, new_col = row + dr, col + dc
                
                if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size:
                    tentative_g = current_g + cost_board[new_row, new_col]
                    
                    if tentative_g < g_score[new_row, new_col]:
                        g_score[new_row, new_col] = tentative_g
                        f_score = tentative_g + heuristic((new_row, new_col), is_horizontal, cost_board)
                        heapq.heappush(open_set, (f_score, tentative_g, (new_row, new_col)))
        
        return self.board_size * self.board_size

    def minimax(self, board, depth, is_maximizing, alpha, beta, start_time, time_limit):
        # Verificar tiempo más frecuentemente
        if start_time is not None and time_limit is not None:
            if time.time() - start_time > time_limit * 0.9:
                return None
                
        # Verificar victoria/derrota inmediata
        player_distance = self.shortest_path_length(board, 1)
        if player_distance == 0:
            return 1000
        
        opponent_distance = self.shortest_path_length(board, -1)
        if opponent_distance == 0:
            return -1000
        
        # Resto de la función igual, pero añadiendo verificación de tiempo periódica
        
        board_hash = hash(board.tobytes())
        
        if board_hash in self.transposition_table and self.transposition_table[board_hash]['depth'] >= depth:
            return self.transposition_table[board_hash]['value']
        
        if depth == 0:
            value = self.evaluate(board)
            self.transposition_table[board_hash] = {'value': value, 'depth': depth}
            return value
        
        empty_cells = self.get_ordered_moves(board)
        
        if is_maximizing:
            value = float('-inf')
            for cell in empty_cells:
                # Verificar tiempo periódicamente
                if start_time is not None and time_limit is not None and time.time() - start_time > time_limit:
                    return None
                
                move = cell[0] * self.board_size + cell[1]
                eval_value = self.minimax(
                    self.make_move(board, move, 1), 
                    depth - 1, False, alpha, beta, start_time, time_limit
                )
                
                # Si una búsqueda más profunda se quedó sin tiempo
                if eval_value is None:
                    return None
                
                value = max(value, eval_value)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
        else:
            value = float('inf')
            for cell in empty_cells:
                # Verificar tiempo periódicamente
                if start_time is not None and time_limit is not None and time.time() - start_time > time_limit:
                    return None
                
                move = cell[0] * self.board_size + cell[1]
                eval_value = self.minimax(
                    self.make_move(board, move, -1), 
                    depth - 1, True, alpha, beta, start_time, time_limit
                )
                
                # Si una búsqueda más profunda se quedó sin tiempo
                if eval_value is None:
                    return None
                
                value = min(value, eval_value)
                beta = min(beta, value)
                if beta <= alpha:
                    break
        
        self.transposition_table[board_hash] = {'value': value, 'depth': depth}
        return value

    def find_best_move_with_time(self, board, start_time, remaining_time):
        time_limit = remaining_time * 0.9  # 90% del tiempo restante para tener margen
        
        # Primera jugada: elegir centro si es posible
        if np.sum(board != 0) <= 1:
            center = self.board_size * self.board_size // 2
            row, col = divmod(center, self.board_size)
            if board[row, col] == 0:
                return center
        
        # Inicialización
        best_move = None
        best_value = float('-inf')
        previous_depth_best_move = None
        
        # Orden inicial de movimientos
        ordered_cells = self.get_ordered_moves(board)
        
        # Si hay movimientos disponibles, selecciona el primero como respaldo
        if ordered_cells:
            fallback_move = ordered_cells[0][0] * self.board_size + ordered_cells[0][1]
        else:
            free = np.flatnonzero(board == 0)
            if free.size > 0:
                fallback_move = np.random.choice(free)
            else:
                return 0
        
        # Reducir max_depth si hay poco tiempo
        current_time = time.time() - start_time
        max_depth = self.max_depth
        if current_time > time_limit * 0.3:
            max_depth = min(max_depth, 3)  # Reducir profundidad si ya gastamos 30% del tiempo
        
        # Profundización iterativa con verificaciones de tiempo agresivas
        for depth in range(1, max_depth + 1):
            # Verificar si vale la pena iniciar esta profundidad
            current_time = time.time() - start_time
            if current_time > time_limit * 0.7:  # Si ya usamos 70% del tiempo
                break  # No iniciar una nueva profundidad
                
            depth_best_move = None
            depth_best_value = float('-inf')
            alpha = float('-inf')
            
            # Explorar movimientos ordenados
            for cell in ordered_cells:
                # Verificar tiempo de forma más agresiva
                current_time = time.time() - start_time
                if current_time > time_limit * 0.8:  # 80% del límite
                    # Salir antes para garantizar una respuesta
                    return best_move if best_move is not None else fallback_move
                
                move = cell[0] * self.board_size + cell[1]
                board_copy = self.make_move(board, move, 1)
                move_value = self.minimax(board_copy, depth - 1, False, alpha, float('inf'), start_time, time_limit)
                
                # Si el minimax indica que se acabó el tiempo
                if move_value is None:
                    # Verificar que tenemos un buen fallback
                    if best_move is None and depth > 1:
                        # Usar el mejor movimiento de la profundidad anterior
                        best_move = previous_depth_best_move
                    
                    # Último recurso: elegir el movimiento con la mayor conectividad
                    if best_move is None:
                        best_move = self.get_most_connected_move(board, ordered_cells)
                    
                    return best_move if best_move is not None else fallback_move
                
                if move_value > depth_best_value:
                    depth_best_value = move_value
                    depth_best_move = move
                
                alpha = max(alpha, depth_best_value)
            
            # Para el final del juego, enfatizar la conectividad
            empty_count = np.sum(board == 0)
            late_game = empty_count < self.board_size * 2
            if late_game:
                # Modificar ordered_cells para priorizar conectividad
                ordered_cells = self.get_endgame_moves(board, ordered_cells)
            
            # Actualizar el mejor movimiento encontrado hasta ahora
            if depth_best_move is not None:
                best_move = depth_best_move
                best_value = depth_best_value
                previous_depth_best_move = depth_best_move
        
        # Retornar el mejor movimiento encontrado
        return best_move if best_move is not None else fallback_move

    def get_ordered_moves(self, board):
        """
        Ordena los movimientos posibles para mejorar la eficiencia de poda alfa-beta.

        Criterios de ordenamiento (menor score es mejor):
        - Distancia al centro
        - Bonus por estar en bordes estratégicos
        - Bonus por continuidad con piezas existentes
        - Bonus por posiciones que reducen significativamente la longitud del camino
        """
        empty_cells = self.get_empty_cells(board)
        scored_moves = []
        
        for cell in empty_cells:
            row, col = cell
            move = row * self.board_size + col
            
            center_dist = abs(row - self.board_size//2) + abs(col - self.board_size//2)
            edge_bonus = 0
            if row == 0 or row == self.board_size-1 or col == 0 or col == self.board_size-1:
                edge_bonus = 2
                
            # preferir movimientos cerca de piezas existentes
            continuity_bonus = self.calculate_continuity(board, row, col)
            
            is_connected = self.is_connected_to_path(board, row, col)
            connectivity_factor = 0 if is_connected else 200  # Mucho más alto para garantizar conexión
            
            # Nueva fórmula que incluye conectividad como factor crítico
            score = connectivity_factor + center_dist - edge_bonus - (continuity_bonus * 1.5)
            scored_moves.append((score, move, cell))
        
        scored_moves.sort()
        return [m[2] for m in scored_moves]

    def is_connected_to_path(self, board, row, col):
        """Verifica si una posición está conectada al camino existente"""
        # Si es la primera jugada, cualquier posición es válida
        our_pieces = np.where(board == 1)
        if len(our_pieces[0]) == 0:
            return True
        
        # Verificar adyacencia directa
        for dr, dc in self.directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == 1:
                return True
        
        return False

    def calculate_continuity(self, board, row, col):
        bonus = 0
        
        # Evaluar dirección estratégica
        if self.is_strategic_position(board, row, col, 1):
            bonus += 10 
        
        # Resto del código para bonificación por adyacencia
        for dr, dc in self.directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if board[nr, nc] == 1:  # Si hay una pieza propia adyacente
                    bonus += 2
                elif board[nr, nc] == -1:  # Si hay una pieza enemiga adyacente
                    bonus += 1  # Menor bonus pero sigue siendo relevante
        
        
        horizontal_progress = 0
        our_pieces = np.where(board == 1)
        if len(our_pieces[0]) > 0:
            leftmost_col = min(our_pieces[1])
            rightmost_col = max(our_pieces[1])
            
            if col < leftmost_col or col > rightmost_col:
                horizontal_progress = 10
                
        bonus += horizontal_progress
        
        return bonus

    def is_strategic_position(self, board, row, col, player):
        """
        Determina si un movimiento es estratégicamente valioso al reducir
        significativamente la longitud del camino necesario para ganar.
        """
        # Crear tablero temporal con este movimiento
        temp_board = board.copy()
        temp_board[row, col] = player
        
        # Calcular distancia antes y después de este movimiento
        original_dist = self.shortest_path_length(board, player)
        new_dist = self.shortest_path_length(temp_board, player)
        return new_dist < original_dist - 1

    def get_endgame_moves(self, board, ordered_cells):
        """Reordena los movimientos para el final del juego priorizando completamente la conectividad"""
        scored_endgame = []
        
        # Buscar nuestras piezas en el tablero
        our_pieces = np.where(board == 1)
        our_positions = list(zip(our_pieces[0], our_pieces[1]))
        
        # Para cada movimiento posible, evaluar si conecta con nuestras piezas
        for cell in ordered_cells:
            row, col = cell
            
            # Calcular distancia mínima a nuestras piezas (para priorizar continuidad)
            min_distance = self.board_size * 2
            for p_row, p_col in our_positions:
                dist = abs(row - p_row) + abs(col - p_col)
                min_distance = min(min_distance, dist)
            
            scored_endgame.append((min_distance, cell))
        
        # Ordenar por distancia mínima (menor es mejor)
        scored_endgame.sort()
        return [m[1] for m in scored_endgame]

    def get_most_connected_move(self, board, ordered_cells):
        """Encuentra el movimiento con mayor conectividad"""
        max_connectivity = float('-inf')
        best_move = None
        
        for cell in ordered_cells:
            row, col = cell
            connectivity = self.calculate_continuity(board, row, col)
            if connectivity > max_connectivity:
                max_connectivity = connectivity
                best_move = row * self.board_size + col
        
        return best_move

    def check_potential_win(self, board, player, start_time=None, time_limit=None):
        """Versión con control de tiempo"""
        for cell in self.get_empty_cells(board):
            # Verificar tiempo
            if start_time is not None and time_limit is not None:
                if time.time() - start_time > time_limit * 0.9:  # 90% del límite
                    return None  # Abandonar la búsqueda si se acerca al límite
                    
            row, col = cell
            test_board = board.copy()
            test_board[row, col] = player
            
            # Usar la función rápida has_connecting_path en lugar de shortest_path_length
            if self.has_connecting_path(test_board, player):
                return row * self.board_size + col
        
        return None

    def get_fast_move(self, board):
        """Devuelve un movimiento rápido cuando hay poco tiempo"""
        # Obtener movimientos ordenados de forma ligera
        empty_cells = self.get_empty_cells(board)
        
        # Priorizar movimientos conectados a nuestras piezas
        for cell in empty_cells:
            row, col = cell
            if self.is_connected_to_path(board, row, col):
                return row * self.board_size + col
        
        # Si no hay movimientos conectados, elegir el primero disponible
        if len(empty_cells) > 0:
            row, col = empty_cells[0]
            return row * self.board_size + col
        
        # Última opción: aleatorio
        free = np.flatnonzero(board == 0)
        if free.size > 0:
            return np.random.choice(free)
        return 0