import streamlit as st
from PIL import Image

import style

st.title('PyTorch Style Transfer')

# Allow the user to upload a file (image)
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

style_name = st.sidebar.selectbox(
    'Select Style',
    ('candy', 'mosaic', 'rain_princess', 'udnie')
)


model = "saved_models/" + style_name + ".pth"

# Process the uploaded file
if uploaded_file is not None:
    # Save the uploaded file temporarily
    image_path = "temp_content_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Display the selected image
    st.image(image_path, caption="Uploaded Image", use_column_width=True)

    # Stylize the image when the user clicks the button
    clicked = st.button('Stylize')

    if clicked:
        model = style.load_model(model)
        output_image_path = "temp_output_image.jpg"
        style.stylize(model, image_path, output_image_path)

        st.write('### Output image:')
        output_image = Image.open(output_image_path)
        st.image(output_image, caption="Stylized Image", use_column_width=True)
else:
    st.warning("Please upload an image.")
