from PIL import Image, ImageDraw
import numpy as np

def create_icon(name, size=(32, 32), color=(255, 255, 255)):
    # Create a new image with a dark background
    image = Image.new('RGBA', size, (54, 54, 54, 255))
    draw = ImageDraw.Draw(image)

    # Draw a simple shape based on the icon type
    if name == 'roddier':
        # Draw a circle for Roddier test with thicker border
        margin = 4
        draw.ellipse([margin, margin, size[0]-margin, size[1]-margin],
                    outline=color, width=2)
        # Add a dot in the center
        center = size[0] // 2
        dot_size = 4
        draw.ellipse([center-dot_size, center-dot_size,
                     center+dot_size, center+dot_size],
                    fill=color)
    elif name == 'interferometry':
        # Draw wave pattern for interferometry with thicker lines
        margin = 4
        height = size[1] // 2
        points = []
        for x in range(0, size[0], 4):
            y = height + int((size[1]/4) * (-1 if (x//4) % 2 else 1))
            points.append((x, y))
        if points:
            # Draw multiple offset lines for a thicker appearance
            for offset in [-1, 0, 1]:
                offset_points = [(x, y + offset) for x, y in points]
                draw.line(offset_points, fill=color, width=2)
    elif name == 'trash':
        # Draw a trash can icon
        margin = 4
        # Draw the main body of the trash can
        draw.rectangle([margin, margin+8, size[0]-margin, size[1]-margin],
                      outline=color, width=2)
        # Draw the top handle
        handle_width = 8
        handle_height = 4
        handle_x = (size[0] - handle_width) // 2
        draw.rectangle([handle_x, margin, handle_x+handle_width, margin+handle_height],
                      outline=color, width=2)
        # Draw the lid
        lid_width = 12
        lid_height = 2
        lid_x = (size[0] - lid_width) // 2
        draw.rectangle([lid_x, margin+handle_height, lid_x+lid_width, margin+handle_height+lid_height],
                      outline=color, width=2)
    elif name == 'theme':
        # Draw a sun/moon icon for theme toggle
        margin = 4
        center = size[0] // 2
        radius = (size[0] - 2*margin) // 2
        # Draw the main circle
        draw.ellipse([margin, margin, size[0]-margin, size[1]-margin],
                    outline=color, width=2)
        # Draw rays
        for i in range(8):
            angle = i * 45
            ray_length = radius - 4
            start_x = center + int(ray_length * 0.3 * np.cos(np.radians(angle)))
            start_y = center + int(ray_length * 0.3 * np.sin(np.radians(angle)))
            end_x = center + int(ray_length * np.cos(np.radians(angle)))
            end_y = center + int(ray_length * np.sin(np.radians(angle)))
            draw.line([start_x, start_y, end_x, end_y], fill=color, width=2)

    # Save the icon
    image.save(f'icons/{name}.png')

if __name__ == '__main__':
    # Create all icons
    create_icon('roddier')
    create_icon('interferometry')
    create_icon('trash')
    create_icon('theme')