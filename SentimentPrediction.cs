using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;

namespace mlnet
{
    partial class Program
    {
        public class SentimentPrediction
        {
            [ColumnName("PredictedLabel")]
            public DvBool Sentiment;
        }
    }
}
