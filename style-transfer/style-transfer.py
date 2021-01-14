from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

base_image_path = "style-transfer/datasets/nastia.png"
style_reference_image_path = tf.keras.utils.get_file(
    "starry_night.jpg", "https://i.imgur.com/9ooB60I.jpg"
)
comb_image_path = "nastiia-noize-v3.png"
#base_img = image.load_img(base_image_path)
#style_img = image.load_img(style_reference_image_path)

width, height = image.load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)
style_weight = 1e-6
content_weight = 8e-6
total_variation_weight = 1e-6

model = VGG19(weights='imagenet', include_top=False)
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

#styles are better catched by first layers, with low-level features
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

content_layer_name = "block5_conv2"

feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=outputs_dict)

#base_model = VGG19(weights='imagenet')
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
#img_path = 'style-transfer/datasets/car.jpg'
#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)

def preprocess_image(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return tf.convert_to_tensor(img)

def load_and_preprocess_image(image_path):
    # Util function to open, resize and format pictures into appropriate tensors
    img = image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )

    return preprocess_image(img)

def deprocess_image(x):
    # Util function to convert a tensor into a valid image
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

def gram_matrix(x):
    # make channels first
    x = x[0, :, :, :]
    x = tf.transpose(x, (2, 0, 1))
    # unroll channels X width X height into c X (w * h)
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    # c X c matrix
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))

def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))

def compute_loss(combination_image, base_image_features, style_reference_features, images):

    #TODO: why are we calculating features for reference and styles? they are constants 
    combination_features = feature_extractor(combination_image)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    base_layer_features = base_image_features[content_layer_name]
    combination_layer_features = combination_features[content_layer_name]
    loss = loss + content_weight * content_loss(
        base_layer_features, combination_layer_features
    )
    # Add style loss
    for layer_name in style_layer_names:
        style_layer_features = style_reference_features[layer_name]
        combination_layer_features = combination_features[layer_name]
        sl = style_loss(style_layer_features, combination_layer_features)
        loss += (style_weight / len(style_layer_names)) * sl

    loss += total_variation_weight * total_variation_loss(images)

    return loss

def compute_loss_and_grads(combination_image, base_image_features, style_reference_features, images):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image_features, style_reference_features, images)
    grads = tape.gradient(loss, combination_image)
    
    return loss, grads

optimizer = tf.keras.optimizers.SGD(
    tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=200.0, decay_steps=100, decay_rate=0.98
    )
)

base_image = load_and_preprocess_image(base_image_path)
style_reference_image = load_and_preprocess_image(style_reference_image_path)

combination_image = tf.Variable(load_and_preprocess_image(comb_image_path))
#img = np.random.random(size=(img_nrows, img_ncols, 3))
#pil_img = tf.keras.preprocessing.image.array_to_img(img)

#combination_image = tf.Variable(preprocess_image(pil_img))


base_image_features = feature_extractor(base_image)
style_reference_features = feature_extractor(style_reference_image)

iterations = 5000
result_prefix = 'nastiia-noize-v3'
for i in range(0, iterations + 1):
    images = tf.concat((combination_image, base_image, style_reference_image), axis=0)

    loss, grads = compute_loss_and_grads(
        combination_image, base_image_features, style_reference_features, images
    )
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 10 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))
    if i % 50 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))
        img = deprocess_image(combination_image.numpy())
        fname = result_prefix + "_at_iteration_%d.png" % i
        tf.keras.preprocessing.image.save_img(fname, img)
#preds = base_model.predict(x)
#print('Predicted:', decode_predictions(preds, top=3))
