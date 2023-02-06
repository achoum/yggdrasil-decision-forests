#include <stdio.h>
#include <time.h>
#include <windows.h>

#include "dll.h"

#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace ygg = yggdrasil_decision_forests;

void run_ydf() {
  // Enable the logging. Optional in most cases.

  // Flags
  const std::string dataset_dir =
      "../../yggdrasil_decision_forests/test_data/dataset/";
  const std::string output_dir = "result";

  // Path to the training and testing dataset.
  const auto train_dataset_path =
      absl::StrCat("csv:", dataset_dir, "adult_train.csv");

  const auto test_dataset_path =
      absl::StrCat("csv:", dataset_dir, "adult_test.csv");

  // Create the output directory
  QCHECK_OK(file::RecursivelyCreateDir(output_dir, file::Defaults()));

  // Scan the columns of the dataset to create a dataspec.
  // Same as :infer_dataspec
  YDF_LOG(INFO) << "Create dataspec";
  const auto dataspec_path = file::JoinPath(output_dir, "dataspec.pbtxt");
  ygg::dataset::proto::DataSpecification dataspec;
  ygg::dataset::CreateDataSpec(train_dataset_path, false, /*guide=*/{},
                               &dataspec);
  QCHECK_OK(file::SetTextProto(dataspec_path, dataspec, file::Defaults()));

  // Display the dataspec in a human readable form.
  // Same as :show_dataspec
  YDF_LOG(INFO) << "Nice print of the dataspec";
  const auto dataspec_report =
      ygg::dataset::PrintHumanReadable(dataspec, false);
  QCHECK_OK(
      file::SetContent(absl::StrCat(dataspec_path, ".txt"), dataspec_report));
  YDF_LOG(INFO) << "Dataspec:\n" << dataspec_report;

  // Train the model.
  // Same as :train
  YDF_LOG(INFO) << "Train model";

  // Configure the learner.
  ygg::model::proto::TrainingConfig train_config;
  train_config.set_learner("RANDOM_FOREST");
  train_config.set_task(ygg::model::proto::Task::CLASSIFICATION);
  train_config.set_label("income");
  std::unique_ptr<ygg::model::AbstractLearner> learner;
  QCHECK_OK(GetLearner(train_config, &learner));

  // Set to export the training logs.
  learner->set_log_directory(output_dir);

  // Effectively train the model.
  auto model = learner->TrainWithStatus(train_dataset_path, dataspec).value();

  // Save the model.
  YDF_LOG(INFO) << "Export the model";
  const auto model_path = file::JoinPath(output_dir, "model");
  QCHECK_OK(ygg::model::SaveModel(model_path, model.get()));

  // Show information about the model.
  // Like :show_model, but without the list of compatible engines.
  std::string model_description;
  model->AppendDescriptionAndStatistics(/*full_definition=*/false,
                                        &model_description);
  QCHECK_OK(
      file::SetContent(absl::StrCat(model_path, ".txt"), model_description));
  YDF_LOG(INFO) << "Model:\n" << model_description;

  // Evaluate the model
  // Same as :evaluate
  ygg::dataset::VerticalDataset test_dataset;
  QCHECK_OK(ygg::dataset::LoadVerticalDataset(
      test_dataset_path, model->data_spec(), &test_dataset));

  ygg::utils::RandomEngine rnd;
  ygg::metric::proto::EvaluationOptions evaluation_options;
  evaluation_options.set_task(model->task());

  // The effective evaluation.
  const ygg::metric::proto::EvaluationResults evaluation =
      model->Evaluate(test_dataset, evaluation_options, &rnd);

  // Export the raw evaluation.
  const auto evaluation_path = file::JoinPath(output_dir, "evaluation.pbtxt");
  QCHECK_OK(file::SetTextProto(evaluation_path, evaluation, file::Defaults()));

  // Export the evaluation to a nice text.
  std::string evaluation_report;
  QCHECK_OK(
      ygg::metric::AppendTextReportWithStatus(evaluation, &evaluation_report));
  QCHECK_OK(file::SetContent(absl::StrCat(evaluation_path, ".txt"),
                             evaluation_report));
  YDF_LOG(INFO) << "Evaluation:\n" << evaluation_report;

  // Compile the model for fast inference.
  const std::unique_ptr<ygg::serving::FastEngine> serving_engine =
      model->BuildFastEngine().value();
  const auto &features = serving_engine->features();

  // Handle to two features.
  const auto age_feature = features.GetNumericalFeatureId("age").value();
  const auto education_feature =
      features.GetCategoricalFeatureId("education").value();

  // Allocate a batch of 5 examples.
  std::unique_ptr<ygg::serving::AbstractExampleSet> examples =
      serving_engine->AllocateExamples(5);

  // Set all the values as missing. This is only necessary if you don't set all
  // the feature values manually e.g. SetNumerical.
  examples->FillMissing(features);

  // Set the value of "age" and "eduction" for the first example.
  examples->SetNumerical(/*example_idx=*/0, age_feature, 35.f, features);
  examples->SetCategorical(/*example_idx=*/0, education_feature, "HS-grad",
                           features);

  // Run the predictions on the first two examples.
  std::vector<float> batch_of_predictions;
  serving_engine->Predict(*examples, 2, &batch_of_predictions);

  YDF_LOG(INFO) << "Predictions:";
  for (const float prediction : batch_of_predictions) {
    YDF_LOG(INFO) << "\t" << prediction;
  }
}

DLLEXPORT void hello() { printf("Hello"); }
