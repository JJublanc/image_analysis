import numpy as np
from tensorflow.keras.optimizers import Adam
import sys
from src.model_compile import get_classes, get_anchors, get_initial_stage_and_epoch, \
                              create_model, data_generator_wrapper


def main(data_kind,
         nb_epochs,
         max_layer,
         weights_file_name="yolov3_cards_weights_train_stage-0-epoch-0_.h5"):

    ##############
    # Parameters #
    ##############
    annotation_path = 'data/cards/annotation_{}.txt'.format(data_kind)
    print(annotation_path)
    log_dir = 'load_and_convert_weights/weights/'

    classes_path = 'data/cards/cards.names'
    # classes_path = 'data/coco/coco_names.txt'

    object_detected = classes_path.split("/")[1]
    anchors_path = 'data/cards/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    print("num lignes dans le fichier annotations small",len(lines))
    num_train = len(lines) - num_val
    print(num_train)
    input_shape = (416, 416)
    initial_stage, initial_epoch = get_initial_stage_and_epoch(weights_file_name)

    ################
    # Create model #
    ################
    print(weights_file_name)
    model = create_model(input_shape,
                         anchors,
                         num_classes,
                         freeze_body=2,
                         weights_path='./load_and_convert_weights/weights/{}'.format(weights_file_name))

    ###########################################
    # Train and progressively defreeze layers #
    ###########################################
    for i in range(initial_stage, max_layer + 1):
        print(i)
        print(max_layer)
        model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4),
                  loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change

        print('Unfreeze {} layers and train.'.format(i))
        print(np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables]))
        batch_size = 32  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
             data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
             steps_per_epoch=max(1, num_train // batch_size),
             validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                    num_classes),
             validation_steps=max(1, num_val // batch_size),
             epochs=nb_epochs,
             initial_epoch=initial_epoch)
        # callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'yolov3_{}_weights_{}_stage-{}-epoch-{}_.h5'.format(object_detected, data_kind, i + 1, nb_epochs))


if __name__ == "__main__":
    data_kind = sys.argv[1]
    nb_epochs = int(sys.argv[2])
    max_layer = int(sys.argv[3])
    weights_file_name = sys.argv[4]
    main(data_kind, nb_epochs, max_layer, weights_file_name)

