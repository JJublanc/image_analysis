import numpy as np
import sys

sys.path.append("./")
from load_and_convert_weights.yolov3 import YOLOv3Net
from load_and_convert_weights.yolov3 import parse_cfg


def load_weights(model, cfgfile, weightfile):
    fp = open(weightfile, "rb")
    # The first 5 values are header information
    np.fromfile(fp, dtype=np.int32, count=5)
    blocks = parse_cfg(cfgfile)

    for i, block in enumerate(blocks[1:]):
        if block["type"] == "convolutional":
            conv_layer = model.get_layer('conv_' + str(i))
            print("layer: ", i + 1, conv_layer)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]
            if "batch_normalize" in block:
                norm_layer = model.get_layer('bnorm_' + str(i))
                print("layer: ", i + 1, norm_layer)
                size = np.prod(norm_layer.get_weights()[0].shape)
                bn_weights = np.fromfile(fp, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            else:
                conv_bias = np.fromfile(fp, dtype=np.float32, count=filters)
            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(
                fp, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])
            if "batch_normalize" in block:
                norm_layer.set_weights(bn_weights)
                conv_layer.set_weights([conv_weights])
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

    assert len(fp.read()) == 0, 'failed to read all data'
    fp.close()


def main(num_classes=52,
         filepath='./load_and_convert_weights/weights/yolov3_cards_weights_train_stage-0-epoch-0_.h5'):
    weightfile = "./load_and_convert_weights/weights/yolov3.weights"
    cfgfile = "./load_and_convert_weights/cfg/yolov3.cfg"
    model_size = (416, 416, 3)

    model = YOLOv3Net(cfgfile, model_size, num_classes)
    load_weights(model, cfgfile, weightfile)
    try:
        model.save_weights(filepath=filepath)
        print('\nThe file \'{}\' has been saved successfully.'.format(filepath))
    except IOError:
        print("Couldn't write the file \'yolov3_weights.tf\'.")


if __name__ == '__main__':
    num_class = int(sys.argv[1])
    filepath = sys.argv[2]
    main(num_class, filepath)
