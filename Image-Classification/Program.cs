
/*
 * Name: Dmitry Landy
 * Date: 2/27/2021
 */
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

//specifying the asset path


namespace TransferLearningTF
{
    class Program
    {
        //specifying the asset path
        static readonly string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
        static readonly string _imagesFolder = Path.Combine(_assetsPath, "images");
        static readonly string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
        static readonly string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
        static readonly string _predictSingleImage = Path.Combine(_imagesFolder, "toaster3.jpg");
        static readonly string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(); //starting point for ML.NET operations
            ITransformer model = GenerateModel(mlContext);
            ClassifySingleImage(mlContext, model);
        }

        //maps parameters for the inception model.
        private struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }

        

        //Loads the image, processes it and then makes a prediction.
        public static void ClassifySingleImage(MLContext mlContext, ITransformer model)
        {
            var imageData = new ImageData()
            {
                ImagePath = _predictSingleImage
            };

            // Make prediction function (input = ImageData, output = ImagePrediction)
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictor.Predict(imageData);

            Console.WriteLine($"" +
                $"Image: {Path.GetFileName(imageData.ImagePath)} " +
                $"predicted as: {prediction.PredictedLabelValue} " +
                $"with score: {prediction.Score.Max()} ");
        }

        //The Model Pipeline. This object is what actually breaks down the images in order to 
        //figure out how to classify it with its labels.
        public static ITransformer GenerateModel(MLContext mlContext)
        {
            IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
                // The image transforms transform the images into the model's expected format.
                .Append(mlContext.Transforms.ResizeImages(
                    outputColumnName: "input",
                    imageWidth: InceptionSettings.ImageWidth,
                    imageHeight: InceptionSettings.ImageHeight,
                    inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(
                    outputColumnName: "input",
                    interleavePixelColors: InceptionSettings.ChannelsLast,
                    offsetImage: InceptionSettings.Mean))
                //Data is passed into the TensorFlow model to be processed via deep learning
                .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel).ScoreTensorFlowModel(
                    outputColumnNames: new[] { "softmax2_pre_activation" },
                    inputColumnNames: new[] { "input" },
                    addBatchDimensionInput: true))
                .Append(mlContext.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "LabelKey",
                    inputColumnName: "Label"))
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                    labelColumnName: "LabelKey",
                    featureColumnName: "softmax2_pre_activation"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                    .AppendCacheCheckpoint(mlContext);

            //Preps the training data (tabular data which is numeric and text
            IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);

            //loads the training data into the model
            ITransformer model = pipeline.Fit(trainingData);
            IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);
            IDataView predictions = model.Transform(testData);

            // Create an IEnumerable for the predictions for displaying results
            IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
            DisplayResults(imagePredictionData);

            //Asseses the model from the predicted to the test and returns the model performance metrics
            MulticlassClassificationMetrics metrics =
                mlContext.MulticlassClassification.Evaluate(predictions,
                labelColumnName: "LabelKey",
                predictedLabelColumnName: "PredictedLabel");
            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: " +
                $"{String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

            return model;
        }

        //Displays the results of each prediction.
        private static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
        {
            foreach (ImagePrediction prediction in imagePredictionData)
            {
                Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
            }
        }

        //parses through tags.tsv to get the ImagePath property, load it and the Label into an ImageData object
        public static IEnumerable<ImageData> ReadFromTsv(string file, string folder)
        {
            return File.ReadAllLines(file)
             .Select(line => line.Split('\t'))
             .Select(line => new ImageData(){ ImagePath = Path.Combine(folder, line[0]) });
        }
    }

    //class for Input Data, which is the image
    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath; //file name

        [LoadColumn(1)]
        public string Label; //value of image
    }

    //class for Predictions
    public class ImagePrediction : ImageData
    {
        public float[] Score; //confidence

        public string PredictedLabelValue; //new label based on confidence
    }
}
