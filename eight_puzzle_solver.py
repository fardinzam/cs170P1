import copy, heapq
from typing import Callable, List

BOARD_SIZE_I = 3
BOARD_SIZE_J = 3

EASY_1 = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
EASY_2 = [[1, 2, 3], [4, 5, 6], [7, 0, 8]]
MEDIUM = [[1, 2, 0], [4, 5, 3], [7, 8, 6]]
HARD_1 = [[0, 1, 2], [4, 5, 3], [7, 8, 6]]
HARD_2 = [[8, 7, 1], [6, 0, 2], [5, 4, 3]]

EIGHT_GOAL_STATE = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

SHOW_STEPS = True
NUM_TO_CORRECT_INDEX = {
	1: [0, 0],
	2: [0, 1],
	3: [0, 2],
	4: [1, 0],
	5: [1, 1],
	6: [1, 2],
	7: [2, 0],
	8: [2, 1],
	0: [2, 2]
}


def flatmap(nested_list: List[List]):
	return [item for sublist in nested_list for item in sublist]


############################
# Treenode
############################
class TreeNode:
	def __init__(
		self, 
		state: List[List[int]], 
		parent=None, 
		weight: int=0, 
		g_n: int=0, 
		h_n: int=0, 
		size_i: int=BOARD_SIZE_I, 
		size_j: int=BOARD_SIZE_J
	):
		self.list_state = state
		self.str_state = str(self.list_state)
		self.parent = parent
		self.weight = weight
		self.size_i = size_i
		self.size_j = size_j
		self.g_n = g_n
		self.h_n = h_n

	def num_state_to_str(self):
		return self.str_state

	def is_end_node(self):
		return self.list_state == EIGHT_GOAL_STATE
	
	def __lt__(self, other): ## for heapq
		return self.weight < other.weight

	def generate_adjacent_nodes(self, weight_func: Callable):
		# swap the 0 with adjacent values to generate adjacent nodes
		zero_index_i, zero_index_j = -1, -1
		for i in range(self.size_i):
			for j in range(self.size_j):
				if self.list_state[i][j] == 0:
					zero_index_i = i
					zero_index_j = j
					break
			if zero_index_i != -1 and zero_index_j != -1:
				break
		
		def create_node(i: int, j: int):
			if i < 0 or i >= self.size_i or\
				j < 0 or j >= self.size_j:
				return None
			new_state = copy.deepcopy(self.list_state)
			new_state[zero_index_i][zero_index_j] = new_state[i][j]
			new_state[i][j] = 0
			g_n, h_n, new_node_weight = weight_func(new_state, self.weight)
			return TreeNode(new_state, parent=self, weight=new_node_weight, g_n=g_n, h_n=h_n)

		new_nodes = list(filter(lambda n: n != None, [
			create_node(zero_index_i - 1, zero_index_j),
			create_node(zero_index_i + 1, zero_index_j),
			create_node(zero_index_i, zero_index_j + 1),
			create_node(zero_index_i, zero_index_j - 1),
		]))
		return new_nodes
	
	def __str__(self):
		return '\n'.join([f"[{','.join([str(i) for i in row])}]" for row in self.list_state])


############################
# Priority queue
############################
class PriorityQueue:
    def __init__(self, key: Callable):
        self.heap = []
        self.elements = set()
        self.key = key

    def push(self, val):
        if val not in self.elements:
            heapq.heappush(self.heap, (self.key(val), val))
            self.elements.add(val)

    def pop(self):
        if self.heap:
            _, val = heapq.heappop(self.heap)
            self.elements.remove(val)
            return val
        raise IndexError("Attempted to pop from an empty heap.")

    def push_multiple(self, vals: List):
        for val in vals:
            self.push(val)

    def __len__(self):
        return len(self.heap)

    def is_empty(self):
        return len(self.heap) == 0


############################
# Solver class
############################
class EightPuzzleSolver:
	def __init__(self, weight_func: Callable):
		self.visited_states = set()
		self.state_to_node_map = {}
		self.priority_queue = PriorityQueue(lambda n: n.weight)
		self.path = []
		self.weight_func = weight_func

	def solve(self, starting_state: List[List[int]]):
		_, h_n, _ = self.weight_func(starting_state, 0)
		start_node = TreeNode(starting_state, h_n=h_n)
		self.state_to_node_map[start_node.str_state] = start_node
		self.priority_queue.push(start_node)
		num_nodes_expanded = 0

		curr_node = start_node
		while not self.priority_queue.is_empty():
			curr_node = self.priority_queue.pop()

			# if this state was already explored before, don't explore it again
			if curr_node.str_state in self.visited_states:
				continue
			else:
				self.visited_states.add(curr_node.str_state)

			self.path.append(curr_node)

			 # if end node has been found, stop
			if curr_node.is_end_node(): 
				return curr_node, num_nodes_expanded  # if problem.GOAL-TEST(node.STATE) succeeds then return node
			else:
				adjacent_nodes = curr_node.generate_adjacent_nodes(self.weight_func)
				num_nodes_expanded += 1
				for i in range(len(adjacent_nodes)):
					# update weights and de-dupe
					str_state = adjacent_nodes[i].str_state
					if str_state in self.state_to_node_map:
						self.state_to_node_map[str_state].weight = min(
							self.state_to_node_map[str_state].weight,
							adjacent_nodes[i].weight
						)
						adjacent_nodes[i] = self.state_to_node_map[str_state]
				self.priority_queue.push_multiple(adjacent_nodes)

		return None, num_nodes_expanded  # if EMPTY(nodes) then return "failure"


############################
# Search/heuristic functions
############################
def uniformCostWeightFunction(state: List[List[int]], parent_weight: int):
	g_n = parent_weight
	h_n = 1
	return g_n, h_n, g_n + h_n

def manhattanDistanceWeightFunction(state: List[List[int]], parent_weight: int):
	h_n = 0
	for i in range(len(state)):
		for j in range(len(state[i])):
			correct_i, correct_j = NUM_TO_CORRECT_INDEX[state[i][j]]
			h_n += abs(correct_i - i) + abs(correct_j - j)
	g_n = parent_weight
	return g_n, h_n, g_n + h_n

def misplacedTileWeightFunction(state: List[List[int]], parent_weight: int):
	misplaced_tiles = 0
	for i in range(len(state)):
		for j in range(len(state[i])):
			if state[i][j] != EIGHT_GOAL_STATE[i][j]:
				misplaced_tiles += 1
	g_n = parent_weight
	h_n = misplaced_tiles
	return g_n, h_n, g_n + h_n

##########################
# User interface functions
##########################
def get_weight_function():
	algorithm = input("Select algorithm. (1) for Uniform Cost Search, (2) for the Misplaced Tile Heuristic, "
                      "or (3) the Manhattan Distance Heuristic.\n")
	if algorithm == "1":
		return uniformCostWeightFunction
	elif algorithm == "2":
		return misplacedTileWeightFunction
	elif algorithm == "3":
		return manhattanDistanceWeightFunction

def get_board():
	puzzle_mode = input("Welcome to Dean's 8-Puzzle Solver. Type '1' to use a default puzzle, or '2' to create your own.\n")
	if puzzle_mode == "1":
		selected_difficulty = input("You wish to use a default puzzle. Please enter a desired difficulty on a scale from 0 to 4.\n")
		selected_difficulty = int(selected_difficulty)
		default_boards = [EASY_1, EASY_2, MEDIUM, HARD_1, HARD_2]
		default_board_names = ['Easy 1', 'Easy 2', 'Medium', 'Hard 1', 'Hard 2']
		if selected_difficulty > len(default_boards):
			print("Invalid selection. Defaulting to 'Easy1'.")
			return EASY_1
		else:
			print(f'Difficulty of {default_board_names[selected_difficulty]} selected.')
			return default_boards[selected_difficulty]
	elif puzzle_mode == "2":
		print("Enter your puzzle, using a zero to represent the blank. " +
              "Please only enter valid 8-puzzles. Enter the puzzle delimiting " +
              "the numbers with a space. RET only when finished.\n")
		puzzle_row_one = list(map(int, input("Enter the first row: ").split()))
		puzzle_row_two = list(map(int, input("Enter the second row: ").split()))
		puzzle_row_three = list(map(int, input("Enter the third row: ").split()))
		user_puzzle = [puzzle_row_one, puzzle_row_two, puzzle_row_three]
	return user_puzzle


######################
# Main
######################
if __name__ == '__main__':
	board, weight_function = get_board(), get_weight_function()
	final_node, num_expanded_nodes = EightPuzzleSolver(weight_function).solve(board)
	# report solution
	node = final_node
	solution_depth = -1
	traceback = []
	while node:
		traceback.append(node)
		node = node.parent
		solution_depth += 1
	traceback = traceback[::-1]
	for node in traceback:
		print(f'The best state to expand with a g(n) = {node.g_n} and h(n) = {node.h_n} is...')
		print(node)
	print(f'Solution depth was {solution_depth}')
	print(f'Number of nodes expanded: {num_expanded_nodes}')
