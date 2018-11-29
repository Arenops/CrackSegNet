from model import *
import skimage.transform as trans

model_name = 'Deeplabv3MAX+FL4'
model = unet3s2()
model.load_weights(model_name + '.hdf5')
test_path="test_imgs"
data_name = os.listdir(test_path)

for i in data_name:
    if 'predict' not in i:
        img = io.imread(os.path.join(test_path, i), as_gray=True)
        img = trans.resize(img, (512, 512), mode='edge')
        img = np.reshape(img, (1,) + img.shape)
        results = model.predict(img, verbose=1)
        results = np.squeeze(results)
        results[results <= 0.5] = 0
        results[results > 0.5] = 1
        filename = i.split('.')[0]
        io.imsave(os.path.join("output", filename+model_name+".png"), results)
