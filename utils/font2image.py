# Uses pillow (you can also use another imaging library if you want)
from PIL import Image, ImageFont, ImageDraw

# Load the font and set the font size to 42
font = ImageFont.truetype('../data/dss/font/Habbakuk.TTF', 42)

# Character mapping for each of the 27 tokens
CHAR_MAP = {'Alef': ')',
            'Ayin': '(',
            'Bet': 'b',
            'Dalet': 'd',
            'Gimel': 'g',
            'He': 'x',
            'Het': 'h',
            'Kaf': 'k',
            'Kaf-final': '\\',
            'Lamed': 'l',
            'Mem': '{',
            'Mem-medial': 'm',
            'Nun-final': '}',
            'Nun-medial': 'n',
            'Pe': 'p',
            'Pe-final': 'v',
            'Qof': 'q',
            'Resh': 'r',
            'Samekh': 's',
            'Shin': '$',
            'Taw': 't',
            'Tet': '+',
            'Tsadi-final': 'j',
            'Tsadi-medial': 'c',
            'Waw': 'w',
            'Yod': 'y',
            'Zayin': 'z'}


def create_image(label, img_size):
    """Returns a grayscale image based on specified label of img_size

    :param label: The label to create an image for
    :param img_size: The size of the desired image
    :return: An image of the specified size with specified character
    """
    if label not in CHAR_MAP:
        raise KeyError('Unknown label!')

    # Create blank image and create a draw interface
    img = Image.new('L', img_size, 255)
    draw = ImageDraw.Draw(img)

    # Get size of the font and draw the token in the center of the blank image
    w, h = font.getsize(CHAR_MAP[label])
    draw.text(((img_size[0] - w) / 2, (img_size[1] - h) / 2), CHAR_MAP[label], 0, font)

    return img


def example():
    """Create a 50x50 image of the Alef token and save it to disk

    To get the raw data cast it to a numpy array
    """
    img = create_image('Alef', (50, 50))
    img.save('example_alef.png')

