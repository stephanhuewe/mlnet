using System;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using TextLoader = Microsoft.ML.Data.TextLoader;

namespace mlnet
{
    partial class Program
    {
        static void Main(string[] args)
        {
            string dataPath = "wikipedia-detox-250-line-data.tsv";
            LearningPipeline pipeline = new LearningPipeline();
            TextLoader testData = new TextLoader(dataPath).CreateFrom<SentimentData>(separator: '\t');

            pipeline.Add(testData);
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
            pipeline.Add(new FastTreeBinaryClassifier());
            PredictionModel<SentimentData, SentimentPrediction> model = pipeline.Train<SentimentData, SentimentPrediction>();
            Console.WriteLine(Environment.NewLine + Environment.NewLine);
            
            while (true)
            {
                Console.WriteLine("Please enter some statement:" + Environment.NewLine);

                // Positive: He is the best, and the article should say that.
                // Negative: Please refrain from adding nonsense to Wikipedia.

                var input = Console.ReadLine();

                SentimentData data = new SentimentData
                {
                    SentimentText = input
                };

                SentimentPrediction prediction = model.Predict(data);
                string text = "Prediction: " + prediction.Sentiment + Environment.NewLine;
                Console.WriteLine(text);
            }
        }
    }
}
