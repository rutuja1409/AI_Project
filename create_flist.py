import os
import pickle
from datadings.reader import MsgpackReader

def create_flist_from_msgpack(msgpack_path, output_flist):
    """
    Reads a msgpack file and creates an flist containing the image file names in the msgpack.

    Args:
        msgpack_path (str): Path to the msgpack file.
        output_flist (str): Path to save the output flist file.
    """
    with open(output_flist, "w") as output_f:
        dataset = MsgpackReader(msgpack_path)

        for sample in dataset:
            # Deserialize the "data" value using pickle
            data = pickle.loads(sample["data"])

            if "image" in data and "image_file_path" in data:
                image_path = str(data["image_file_path"])

                # Get only the image file name from the path
                image_name = os.path.basename(image_path)

                output_f.write(f"{image_name}\n")

                print("Image File Name:", image_name)


if __name__ == "__main__":
    msgpack_file = "/ds/documents/ShabbyPages/validation/512x512.msgpack"
    output_flist = "/netscratch/lahoti/Palette-Image-to-Image-Diffusion-Models/datasets/flist/validate.flist"
    create_flist_from_msgpack(msgpack_file, output_flist)
