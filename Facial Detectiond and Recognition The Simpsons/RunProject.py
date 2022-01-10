from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *


params: Parameters = Parameters()

if params.use_flip_images:
    params.number_positive_examples *= 2

facial_detector: FacialDetector = FacialDetector(params)
#exemple pozitive
positive_features_path = os.path.join(params.dir_save_files, 'descriptoriExemplePozitive_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_positive_examples) + '.npy')
if os.path.exists(positive_features_path):
    positive_features = np.load(positive_features_path)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    positive_features = facial_detector.get_positive_descriptors()
    np.save(positive_features_path, positive_features)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_path)

# exemple negative
negative_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegative_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_negative_examples) + '.npy')
if os.path.exists(negative_features_path):
    negative_features = np.load(negative_features_path)
    print('Am incarcat descriptorii pentru exemplele negative')
else:
    print('Construim descriptorii pentru exemplele negative:')
    negative_features = facial_detector.get_negative_descriptors()
    np.save(negative_features_path, negative_features)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)

#clasificator
training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))
facial_detector.train_classifier(training_examples, train_labels)

detections, scores, file_names, ch_detections, ch_scores, ch_file_names = facial_detector.run()

eval_dir_task1 = "evaluare//fisiere_solutie//Liviu_Bouruc_334//task1"
eval_dir_task2 = "evaluare//fisiere_solutie//Liviu_Bouruc_334//task2"

np.save(os.path.join(eval_dir_task1, 'detections_all_faces.npy'), detections)
np.save(os.path.join(eval_dir_task1, 'scores_all_faces.npy'), scores)
np.save(os.path.join(eval_dir_task1, 'file_names_all_faces.npy'), file_names)

np.save(os.path.join(eval_dir_task2, 'detections_bart.npy'), ch_detections[0])
np.save(os.path.join(eval_dir_task2, 'scores_bart.npy'), ch_scores[0])
np.save(os.path.join(eval_dir_task2, 'file_names_bart.npy'), ch_file_names[0])

np.save(os.path.join(eval_dir_task2, 'detections_homer.npy'), ch_detections[1])
np.save(os.path.join(eval_dir_task2, 'scores_homer.npy'), ch_scores[1])
np.save(os.path.join(eval_dir_task2, 'file_names_homer.npy'), ch_file_names[1])

np.save(os.path.join(eval_dir_task2, 'detections_lisa.npy'), ch_detections[2])
np.save(os.path.join(eval_dir_task2, 'scores_lisa.npy'), ch_scores[2])
np.save(os.path.join(eval_dir_task2, 'file_names_lisa.npy'), ch_file_names[2])

np.save(os.path.join(eval_dir_task2, 'detections_marge.npy'), ch_detections[3])
np.save(os.path.join(eval_dir_task2, 'scores_marge.npy'), ch_scores[3])
np.save(os.path.join(eval_dir_task2, 'file_names_marge.npy'), ch_file_names[3])


if params.has_annotations:
    facial_detector.eval_detections(detections, scores, file_names, params.path_annotations, "")
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)

facial_detector.eval_detections(ch_detections[0], ch_scores[0], ch_file_names[0], 'validare//task2_bart_gt.txt', "Bart ")
facial_detector.eval_detections(ch_detections[1], ch_scores[1], ch_file_names[1], 'validare//task2_homer_gt.txt', "Homer ")
facial_detector.eval_detections(ch_detections[2], ch_scores[2], ch_file_names[2], 'validare//task2_lisa_gt.txt', "Lisa ")
facial_detector.eval_detections(ch_detections[3], ch_scores[3], ch_file_names[3], 'validare//task2_marge_gt.txt', "Marge ")
