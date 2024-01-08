using Emgu.CV.ML.MlEnum;
using Emgu.CV.ML;
using Emgu.CV.Structure;
using Emgu.CV;

namespace NumberPlateRecognition
{
    public class SVMClassification
    {
        public SVM SvmClassifier { get; set; }

        public SVMClassification(string fileName)
        {
            // Učitavanje podataka iz trening modela
            Mat SVM_TrainingData = new Mat();
            Mat SVM_Classes = new Mat();
            using (FileStorage fs = new FileStorage(fileName, FileStorage.Mode.Read))
            {
                fs.GetNode("TrainingData").ReadMat(SVM_TrainingData);
                fs.GetNode("classes").ReadMat(SVM_Classes);
            }

            // SVM klasifikator
            SvmClassifier = new SVM()
            {
                Type = SVM.SvmType.CSvc,
                Degree = 0,
                Gamma = 1,
                Coef0 = 0,
                C = 1,
                Nu = 0,
                P = 0,
                TermCriteria = new MCvTermCriteria(1000, 0.01),
            };

            SvmClassifier.SetKernel(SVM.SvmKernelType.Linear);
            SvmClassifier.Train(SVM_TrainingData, DataLayoutType.RowSample, SVM_Classes);
        }
    }
}
