import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras import Model
import matplotlib.pyplot as plt
# Define the dimensions of the generated image
height = 256  # Adjust this value based on your input image height
width = 256   # Adjust this value based on your input image width

# Load pre-trained VGG19 model
vgg = vgg19.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Content layer and style layers
content_layers = ['block5_conv2'] 
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Function to build the model
def get_model(content_layers, style_layers):
    vgg.trainable = False
    outputs = [vgg.get_layer(layer).output for layer in (content_layers + style_layers)]
    model = Model(inputs=[vgg.input], outputs=outputs)
    return model

# Function to preprocess image
def preprocess_image(image_path):
    img = kp_image.load_img(image_path, target_size=(224, 224))
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# Function to deprocess image
def deprocess_image(processed_img):
    x = processed_img.copy()
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Load content and style images
content_path = 'content.jpg'
style_path = 'style.jpg'
content_image = preprocess_image(content_path)
style_image = preprocess_image(style_path)

# Build the model
model = get_model(content_layers, style_layers)

# # Define content and style representations
# # content_target = model(content_image)['block5_conv2']
# content_target = model(content_image)[4]  # Access activations of the fifth layer (index 4)
# style_targets = [model(style_image)[layer] for layer in style_layers]
# Assuming model(content_image) returns a list-like object containing activations for each layer
# and 'block5_conv2' corresponds to the activations of the fifth layer

# Define layer names
layer_names = ['block5_conv2']

# Assuming model(content_image) returns a list-like object containing activations for each layer
# and 'block5_conv2' corresponds to the activations of the fifth layer

# Replace the original line:
# content_target = model(content_image)['block5_conv2']

# With the modified line using integer index:
content_target_index = layer_names.index('block5_conv2')
content_target = model(content_image)[content_target_index]
# Assuming model(style_image) returns a list-like object containing activations for each layer
# and style_layers is a list containing the names of the layers for style extraction

# Create an empty list to store the style targets
style_targets = []

# Iterate over each layer name in style_layers
for layer_name in style_layers:
    layer_found = False
    # Iterate over the output names of the model to find a match
    for i, output_name in enumerate(model.output_names):
        if layer_name in output_name:
            # If the layer name matches part of the output name, append the activations to style_targets
            style_targets.append(model(style_image)[i])
            layer_found = True
            break
    if not layer_found:
        raise ValueError(f"Layer '{layer_name}' is not in the list of layer names")

# Now, style_targets contains the activations of the layers specified in style_layers


# Define loss functions
def content_loss(content, target):
    return tf.reduce_mean(tf.square(content - target))

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

def style_loss(style, gram_target):
    return tf.reduce_mean(tf.square(gram_matrix(style) - gram_target))

def total_variation_loss(image):
    x_deltas, y_deltas = tf.image.image_gradients(image)
    return tf.reduce_mean(tf.square(x_deltas) + tf.square(y_deltas))

# Define weights for content, style, and total variation loss
content_weight = 1e3
style_weight = 1e-2
total_variation_weight = 1e-4

# Initialize generated image with content image
generated_image = tf.Variable(content_image)

# # Define optimizer
# optimizer = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)
learning_rate = 0.001  # Adjust this value based on your specific task and model
optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)


# Main optimization loop
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = model(image)
        content_features = outputs['block5_conv2']
        style_features = [outputs[layer] for layer in style_layers]
        content_loss_val = content_loss(content_features, content_target)
        style_loss_val = 0
        for style_feature, style_target in zip(style_features, style_targets):
            style_loss_val += style_loss(style_feature, style_target)
        style_loss_val /= len(style_layers)
        tv_loss = total_variation_loss(image)
        total_loss = content_weight * content_loss_val + style_weight * style_loss_val + total_variation_weight * tv_loss

    grad = tape.gradient(total_loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0))



# Define your loss functions
def calculate_content_loss(content_features, generated_features):
    return tf.reduce_mean(tf.square(content_features - generated_features))

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

# def calculate_style_loss(style_targets, generated_features):
#     style_loss = 0
#     for target, gen in zip(style_targets, generated_features):
#         target_gram = gram_matrix(target)
#         gen_gram = gram_matrix(gen)
#         style_loss += tf.reduce_mean(tf.square(target_gram - gen_gram))
#     return style_loss
def calculate_style_loss(style_targets, generated_features):
    style_loss = 0
    for target, gen in zip(style_targets, generated_features):
        target_gram = gram_matrix(target)
        gen_gram = gram_matrix(gen)
        # Compute the mean squared difference between Gram matrices
        style_loss += tf.reduce_mean(tf.square(target_gram - gen_gram))
    return style_loss
# Calculate style loss
    style_loss = calculate_style_loss(style_targets, generated_features)

def calculate_total_variation_loss(image):
    x_diff = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_diff = image[:, 1:, :, :] - image[:, :-1, :, :]
    return tf.reduce_sum(tf.abs(x_diff)) + tf.reduce_sum(tf.abs(y_diff))

# # Define your train_step function
# def train_step(generated_image):
#     with tf.GradientTape() as tape:
#         # Forward pass
#         generated_features = model(generated_image)
        
#         # Extract content features from model output
#         content_features = generated_features[4]  # Assuming 'block5_conv2' is at index 4
        
#         # Calculate content loss
#         content_loss_val = calculate_content_loss(content_features, generated_features[4])
        
#         # Total variation loss
#         total_variation_loss = calculate_total_variation_loss(generated_image)
        
#         # Total loss
#         total_loss = content_loss_val + style_loss_val + total_variation_loss

#     # Compute gradients
#     gradients = tape.gradient(total_loss, generated_image)

#     # Update generated image
#     optimizer.apply_gradients([(gradients, generated_image)])
    
#     return total_loss

# Define your train_step function
def train_step(generated_image, content_target, style_targets):
    with tf.GradientTape() as tape:
        # Forward pass
        outputs = model(generated_image)
        content_features = outputs[0]  # Assuming content layer is the first one
        style_features = outputs[1:]   # Style features start from the second one
        
        # Calculate content loss
        content_loss_val = calculate_content_loss(content_features, content_target)
        
        # Calculate style loss
        style_loss_val = calculate_style_loss(style_targets, style_features)
        
        # Total variation loss
        tv_loss = calculate_total_variation_loss(generated_image)
        
        # Total loss
        total_loss = content_weight * content_loss_val + style_weight * style_loss_val + total_variation_weight * tv_loss

    # Compute gradients
    gradients = tape.gradient(total_loss, generated_image)

    # Update generated image
    optimizer.apply_gradients([(gradients, generated_image)])
    
    return total_loss

    # Initialize the generated image
generated_image = tf.Variable(tf.random.uniform((1, height, width, 3), minval=0, maxval=255.0))

# Optimization loop
for i in range(1000):
    train_step(generated_image,content_target,style_targets)
    if i % 100 == 0:
        print("Iteration:", i)
        



# Initialize the generated image
generated_image = tf.Variable(tf.random.uniform((1, height, width, 3), minval=0, maxval=255.0))


# Plot the result
plt.imshow(deprocess_image(generated_image))
plt.axis('off')
plt.show()
