using Emgu.CV;
using Emgu.CV.UI;
using Emgu.CV.ML;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using System;
using System.Linq;
using System.Drawing;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace NumberPlateRecognition
{
    internal class Program
    {
        static void Main(string[] args)
        {
            bool useSyntax = true;

            ImageViewer viewer = new ImageViewer();
            SVMClassification SVM = new SVMClassification("SVM.xml");
            OCRecognition OCR = new OCRecognition("hrv+eng");

            for (int i = 0; i < 54; i++)
            {
                var imagePath = "C:\\Users\\andrija.milosavljevi\\OneDrive - Coming Computer Engineering\\Desktop\\Master\\SIR1\\Test slike\\seat" + i + ".jpg";
                Image<Bgr, Byte> originalImage = new Image<Bgr, Byte>(imagePath);

                // Segmentacija
                var possibleRegions = DetectRegion.ExtractSegments(originalImage);

                // Klasifikacija
                List<Plate> plates = new List<Plate>();
                foreach (var region in possibleRegions)
                {
                    //viewer.Image = region.PlateImage;
                    //viewer.ShowDialog();

                    Mat img = region.PlateImage;
                    Mat p = img.Reshape(1, 1);
                    p.ConvertTo(p, DepthType.Cv32F);

                    var response = SVM.SvmClassifier.Predict(p);

                    if (response == 1)
                    {
                        plates.Add(region);
                    }
                }

                foreach (var plate in plates)
                {
                    viewer.Image = plate.PlateImage;
                    viewer.ShowDialog();

                    // OCR segmentacija
                    var charSegments = OCR.Segmentation(plate, false);

                    foreach (var segment in charSegments)
                    {
                        // OCR klasifikacija
                        var character = OCR.Classify(segment.CharImage);
                        if (!string.IsNullOrEmpty(character))
                        {
                            plate.Chars.Add(character.Trim().ToUpper().FirstOrDefault());
                            plate.CharsPositions.Add(segment.Position);
                        }
                        else
                        {
                            Rectangle borderRect = new Rectangle(2, 2, segment.CharImage.Width-5, segment.CharImage.Height-5);
                            CvInvoke.Rectangle(segment.CharImage, borderRect, new MCvScalar(0, 0, 0), 1);

                            character = OCR.Classify(segment.CharImage);
                            plate.Chars.Add(character.Trim().ToUpper().FirstOrDefault());
                            plate.CharsPositions.Add(segment.Position);
                        }
                    }

                    // Sortiranje slova, da idu po redosledu
                    for (int j = 0; j <= plate.Chars.Count - 2; j++)
                    {
                        for (int k = 0; k <= plate.Chars.Count - 2; k++)
                        {
                            if (plate.CharsPositions[k].X > plate.CharsPositions[k + 1].X)
                            {
                                var temp = plate.Chars[k + 1];
                                plate.Chars[k + 1] = plate.Chars[k];
                                plate.Chars[k] = temp;

                                var temp1 = plate.CharsPositions[k + 1];
                                plate.CharsPositions[k + 1] = plate.CharsPositions[k];
                                plate.CharsPositions[k] = temp1;
                            }
                        }
                    }

                    string licensePlate = plate.ToString();

                    // Sintaksna analiza
                    if (useSyntax && !string.IsNullOrEmpty(licensePlate) && licensePlate.Length > 2)
                    {
                        var startLetters = licensePlate.Substring(0, 2); // Geografsko podrucje
                        var endLetters = licensePlate.Substring(licensePlate.Length - 2, 2);
                        var numbers = licensePlate.Substring(2, licensePlate.Length - 4);

                        if (!SyntaxHelper.MatchLetters(startLetters))
                        {
                            startLetters = SyntaxHelper.ChangeLetters(startLetters);
                        }

                        if (!SyntaxHelper.MatchLetters(endLetters))
                        {
                            endLetters = SyntaxHelper.ChangeLetters(endLetters);
                        }

                        if (!SyntaxHelper.MatchNumbers(numbers))
                        {
                            numbers = SyntaxHelper.ChangeNumbers(numbers);
                        }

                        licensePlate = startLetters + numbers + endLetters;
                    }

                    if (!string.IsNullOrEmpty(licensePlate) && licensePlate.Length > 2)
                    {
                        CvInvoke.Rectangle(originalImage, plate.Position, new MCvScalar(0, 200, 0), 4);
                        CvInvoke.PutText(originalImage, licensePlate, new Point(plate.Position.X, plate.Position.Y - 30), FontFace.HersheyDuplex, 2, new MCvScalar(0, 200, 0), 8);
                    }
                }

                viewer.Image = originalImage;
                viewer.ShowDialog();
                originalImage.Save("result.png");
            }
        }
    }
}