def generate_movement_instructions(path):
    """
    Generates instructions for moving along a discontinuous path.

    Args:
        path (list): A list of coordinate pairs representing the path, where the first element is the column and the second element is the row.

    Returns:
        list: A list of instructions, similar to the previous version.
    """

    instructions = []

    for i in range(len(path) - 1):
        start_x, start_y = path[i]
        end_x, end_y = path[i + 1]

        # Determine the direction and distance for this segment
        if start_x < end_x and start_y == end_y:  # Move forward
            direction = "forward"
            distance = end_x - start_x
        elif start_x == end_x and start_y < end_y:  # Turn right
            direction = "right"
            distance = end_y - start_y
        elif start_x == end_x and start_y > end_y:  # Turn left
            direction = "left"
            distance = start_y - end_y
        else:
            direction = "right"
            distance = start_x - end_x

        # Move in the determined direction
        while distance > 0:
            instructions.append(("moveForward"))
            distance -= 1

        # Turn if necessary for the next segment
        if i < len(path) - 2:
            next_x, next_y = path[i + 2]
            if direction == "forward" and next_y != end_y:
                instructions.append(("turnRight" if next_y > end_y else "turnLeft"))
            elif direction == "right" and next_x != end_x:
                instructions.append(("turnLeft"))
            elif direction == "left" and next_x != end_x:
                instructions.append(("turnRight"))

    return instructions
# Example usage with a discontinuous path
path = [ (0, 1), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (2, 9), (3, 9), (3, 8), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (10, 7), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13), (12, 13), (13, 13), (14, 13), (15, 13), (15, 12), (15, 11), (16, 11), (17, 11), (17, 12), (17, 13), (17, 14), (17, 15), (17, 16), (17, 17), (17, 18), (17, 19), (18, 19), (19, 19)]
instructions = generate_movement_instructions(path)
print(instructions)

# def generate_movement_instructions(path):
#     """
#     Generates instructions for moving along a discontinuous path, including diagonal movements.

#     Args:
#         path (list): A list of coordinate pairs representing the path, where the first element is the column and the second element is the row.

#     Returns:
#         list: A list of instructions, similar to the previous version.
#     """

#     instructions = []

#     for i in range(len(path) - 1):
#         start_x, start_y = path[i]
#         end_x, end_y = path[i + 1]

#         # Determine the direction and distance for this segment
#         if start_x < end_x and start_y == end_y:  # Move forward
#             direction = "forward"
#             distance = end_x - start_x
#         elif start_x == end_x and start_y < end_y:  # Turn right
#             direction = "right"
#             distance = end_y - start_y
#         elif start_x == end_x and start_y > end_y:  # Turn left
#             direction = "left"
#             distance = start_y - end_y
#         elif start_x < end_x and start_y < end_y:  # Diagonal move - move forward then turn right
#             direction = "forward"
#             distance_x = end_x - start_x
#             distance_y = end_y - start_y
#             while distance_x > 0:
#                 instructions.append(("moveForward"))
#                 distance_x -= 1
#             instructions.append(("turnRight"))
#         elif start_x > end_x and start_y < end_y:  # Diagonal move - turn left then move forward
#             direction = "forward"
#             distance_x = start_x - end_x
#             distance_y = end_y - start_y
#             instructions.append(("turnLeft"))
#             while distance_y > 0:
#                 instructions.append(("moveForward"))
#                 distance_y -= 1
#         elif start_x < end_x and start_y > end_y:  # Diagonal move - move forward then turn left
#             direction = "forward"
#             distance_x = end_x - start_x
#             distance_y = start_y - end_y
#             while distance_x > 0:
#                 instructions.append(("moveForward"))
#                 distance_x -= 1
#             instructions.append(("turnLeft"))
#         elif start_x > end_x and start_y > end_y:  # Diagonal move - turn right then move forward
#             direction = "forward"
#             distance_x = start_x - end_x
#             distance_y = start_y - end_y
#             instructions.append(("turnRight"))
#             while distance_y > 0:
#                 instructions.append(("moveForward"))
#                 distance_y -= 1
        

#         # Move in the determined direction
#         while distance > 0:
#             instructions.append(("moveForward"))
#             distance -= 1

#         # Turn if necessary for the next segment
#         if i < len(path) - 2:
#             next_x, next_y = path[i + 2]
#             if direction == "forward" and next_y != end_y:
#                 instructions.append(("turnRight" if next_y > end_y else "turnLeft"))
#             elif direction == "right" and next_x != end_x:
#                 instructions.append(("turnLeft"))
#             elif direction == "left" and next_x != end_x:
#                 instructions.append(("turnRight"))

#     return instructions

# # Example usage with a discontinuous path
# path = [(0, 1), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (2, 9), (3, 9), (3, 8), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (10, 7), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13), (12, 13), (13, 13), (14, 13), (15, 13), (15, 12), (15, 11), (16, 11), (17, 11), (17, 12), (17, 13), (17, 14), (17, 15), (17, 16), (17, 17), (17, 18), (17, 19), (18, 19), (19, 19)]
# instructions = generate_movement_instructions(path)
# print(instructions)