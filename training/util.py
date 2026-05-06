import mcschematic
import numpy as np

def array_to_schematic(token_array, tok2block, save_path, filename="my_schematic"):
    """
    Converts a 3D numpy array of integer tokens into a Minecraft schematic file.

    Args:
        token_array (np.ndarray): A 3D array where values correspond to block types.
        filename (str): The name of the output schematic file (without extension).
    """
    schem = mcschematic.MCSchematic()

    d, h, w = token_array.shape

    # Iterate through the 3D array and place blocks
    for x in range(d):
        for y in range(h):
            for z in range(w):
                token = token_array[x, y, z]
                # block_name = tok2block.get(f'{token}', "minecraft:air") # Default to air if token not found
                block_name = tok2block[token]
                

                # Place the block in the schematic at the specified coordinates (x, y, z)
                schem.setBlock((x, y, z), block_name)

    # Save the schematic file
    schem.save(save_path, filename, mcschematic.Version.JE_1_21_5, True)
    
    print(f"Successfully saved schematic to {filename}.schem")
