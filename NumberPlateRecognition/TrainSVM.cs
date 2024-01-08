using Emgu.CV;
using Emgu.CV.ML;
using Emgu.CV.CvEnum;

using System;
using System.Collections.Generic;

namespace NumberPlateRecognition
{
    public class TrainSVM
    {
        public static int NumberOfPlates = 40;
        public static int NumberOfNoPlates = 40;

        static void Main(string[] args)
        {
            string pathPlates = "C:\\Users\\andrija.milosavljevi\\source\\repos\\NumberPlateRecognition\\NumberPlateRecognition\\TrainingPicturesSVM\\Plate";
            string pathNoPlates = "C:\\Users\\andrija.milosavljevi\\source\\repos\\NumberPlateRecognition\\NumberPlateRecognition\\TrainingPicturesSVM\\NoPlate";
            int imageWidth = 144;
            int imageHeight = 33;

            using (SVM model = new SVM())
            {
                Mat classes = new Mat(NumberOfPlates + NumberOfNoPlates, 1, DepthType.Cv32F, 1);
                Mat trainingData = new Mat(NumberOfPlates+NumberOfNoPlates, imageWidth*imageHeight, DepthType.Cv32F, 1);

                Mat trainingImages = new Mat();
                List<int> trainingLabels = new List<int>(); 

                for (int i = 1; i <= NumberOfPlates; i++)
                {
                    string imagePath = $"{pathPlates}\\plate{i}.jpg";
                    Mat img = CvInvoke.Imread(imagePath, ImreadModes.Grayscale);
                    img = img.Reshape(1, 1);
                    trainingImages.PushBack(img);
                    trainingLabels.Add(1);
                }

                for (int i = 1; i <= NumberOfNoPlates; i++)
                {
                    string imagePath = $"{pathNoPlates}\\noplate{i}.jpg";
                    Mat img = CvInvoke.Imread(imagePath, ImreadModes.Grayscale);
                    img = img.Reshape(1, 1);
                    trainingImages.PushBack(img);
                    trainingLabels.Add(0);
                }

                trainingImages.CopyTo(trainingData);
                trainingData.ConvertTo(trainingData, DepthType.Cv32F);

                Mat trainingLabelsMat = new Mat(trainingLabels.Count, 1, DepthType.Cv32S, 1);
                trainingLabelsMat.SetTo(trainingLabels.ToArray());
                trainingLabelsMat.CopyTo(classes);

                //model.Train(trainingData, DataLayoutType.RowSample, trainingLabelsMat);

                using (FileStorage fs = new FileStorage("SVM.xml", FileStorage.Mode.Write))
                {
                    fs.Write(trainingData, "TrainingData");
                    fs.Write(classes, "classes");
                }

                Console.WriteLine("Training data saved to SVM.xml");
            }
        }
    }
}
