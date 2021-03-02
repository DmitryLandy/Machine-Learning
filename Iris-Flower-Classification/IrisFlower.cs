/*
 * Name: Dmitry Landy
 * Date: 2/27/2021
 */
using System;
using Microsoft.ML;
using Microsoft.ML.Data;


namespace IrisFlower
{
    class Program
    {

        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            //load data
            IDataView dataView = mlContext.Data.LoadFromTextFile<IrisData>(
                "Data/IRIS.csv",
                hasHeader: false,
                separatorChar: ',');

            //create training pipeline
            var pipeline = mlContext.Transforms.Concatenate(
                "Features",
                "SepalLength",
                "SepalWidth",
                "PetalLength",
                "PetalWidth",
                "Species")

            //K-means clustering algorithm to categorized 3 iris flower types
            .Append(mlContext.Clustering.Trainers.KMeans(
                "Features",
                numberOfClusters: 3));

            //train the model
            Console.WriteLine("Start training model...");
            var model = pipeline.Fit(dataView);
            Console.WriteLine("End of training!");

            //Model Can now be used to predict data
            var predictor = mlContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);
            TestIrisData testingData = new TestIrisData();
            
            var prediction = predictor.Predict(testingData.randIris);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
            //Console.WriteLine($"Species: {prediction.PredictedSpecies}");
            
        }
    }

    class TestIrisData
    {

        private static float sl,sw, pl, pw; //length and width of sepal and petal
        
        internal readonly IrisData randIris = new IrisData
        {
            SepalLength = sl,
            SepalWidth = sw,
            PetalLength = pl,
            PetalWidth = pw
        };
        
        //constructor
        public TestIrisData()
        {
            getValues();
        }

        private static void getValues()
        {
            Console.WriteLine("Testing the model on sample flower\n ");
            Console.Write("Input Sepal Length: ");
            sl = float.Parse(Console.ReadLine());
            Console.Write("Input Sepal Width: ");
            sw = float.Parse(Console.ReadLine());
            Console.Write("Input Petal Length: ");
            pl = float.Parse(Console.ReadLine());
            Console.Write("Input Petal Width: ");
            pw = float.Parse(Console.ReadLine());
        }
        
    }
    

    //This is the input data that will be analyzed: length, width of petal and sepal
    public class IrisData
    {
        [LoadColumn(0)]
        public float SepalLength;

        [LoadColumn(1)]
        public float SepalWidth;

        [LoadColumn(2)]
        public float PetalLength;

        [LoadColumn(3)]
        public float PetalWidth;

        [LoadColumn(4)]
        public string Species;

        
    }

    //This class is for the output
    public class ClusterPrediction
    {
        //ID of the predicted cluster
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        //Array of distances to the custer center. 
        [ColumnName("Score")]
        public float[] Distances;

        //[ColumnName("Species")]
        //public string PredictedSpecies;
        
    }
}
