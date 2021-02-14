/*
 * Name: Dmitry Landy
 * File: taxiFare.cs
 * Date: 2/13/2021
 */
using System;
using TaxiFareML.Model;

namespace taxiFare
{
    class Program
    {
        static void Main(string[] args)
        {
            // Add input data
            var input = new ModelInput();
            input.Trip_distance = 300;
            input.Trip_time_in_secs = 1500;

            // Load model and predict output of sample data
            ModelOutput result = ConsumeModel.Predict(input);
            Console.WriteLine($"Distance: {input.Trip_distance}, " +
                $"Time: {input.Trip_time_in_secs}");
            Console.WriteLine($"Predicted cost: {result.Score}");
        }
    }
}
