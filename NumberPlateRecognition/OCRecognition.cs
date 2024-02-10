using Emgu.CV;
using Emgu.CV.UI;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using System;
using System.Linq;
using System.Drawing;
using System.Collections.Generic;

using Tesseract;
using System.Runtime.InteropServices;

namespace NumberPlateRecognition
{
    public class OCRecognition
    {
        public static string AllCharacters = "0123456789abcdefghijklmnoprstuwxyzABCČĆDĐEFGHIJKLMNOPRSŠTUVWXYZŽ";

        public TesseractEngine ocrEngine { get; set; }

        public OCRecognition() 
        {
        }

        public OCRecognition(string langCode)
        {
            ocrEngine = new TesseractEngine(@"./tessdata", langCode, EngineMode.TesseractAndLstm);
            ocrEngine.SetVariable("tessedit_char_whitelist", AllCharacters); ;
        }

        /// <summary>
        /// Method for OCR segmentation.
        /// </summary>
        /// <param name="plate">Image of detected plate.</param>
        /// <param name="showSegments">Show segments?</param>
        /// <returns>List of segmented characters.</returns>
        public List<CharSegment> Segmentation(Plate plate, bool showSegments = false, bool saveSegments = false)
        {
            var output = new List<CharSegment>();
            var imageViewer = new ImageViewer();

            Mat input = plate.PlateImage;

            // Treshold
            Mat imgTreshold = new Mat();
            CvInvoke.Threshold(input, imgTreshold, 60, 255, ThresholdType.BinaryInv);
            if (saveSegments)
            {
                imgTreshold.Save("OCRtreshold.jpg");
            }

            // Find contours
            Mat imgContours = new Mat();
            imgTreshold.CopyTo(imgContours);

            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            Mat hier = new Mat();
            CvInvoke.FindContours(imgContours, contours, hier, RetrType.External, ChainApproxMethod.ChainApproxNone);

            // Merge contours
            VectorOfVectorOfPoint mergedContours = MergeContours(contours);

            Mat result = new Mat();
            imgTreshold.CopyTo(result);
            CvInvoke.CvtColor(result, result, ColorConversion.Gray2Bgr);

            CvInvoke.DrawContours(result, mergedContours, -1, new MCvScalar(255, 0, 0), 1);

            for (int i = 0; i < mergedContours.Size; i++)
            {
                VectorOfPoint contour = mergedContours[i];

                Rectangle rect = CvInvoke.BoundingRectangle(contour);
                CvInvoke.Rectangle(result, rect, new MCvScalar(0, 255, 0));

                //imageViewer.Image = result;
                //imageViewer.ShowDialog();

                Mat charRect = new Mat(imgTreshold, rect);
                if (VerifySizes(charRect))
                {
                    charRect = PreprocessChar(charRect);

                    // Segmentation for character '0'
                    CvInvoke.FindContours(charRect, contours, hier, RetrType.External, ChainApproxMethod.ChainApproxNone);
                    if (contours.Size > 1)
                    {
                        var structureElement = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(1, 5), new Point(-1, -1));
                        CvInvoke.MorphologyEx(charRect, charRect, MorphOp.Close, structureElement, new Point(-1, -1), 1, BorderType.Default, new MCvScalar());

                        structureElement = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(2, 2), new Point(-1, -1));
                        CvInvoke.MorphologyEx(charRect, charRect, MorphOp.Open, structureElement, new Point(-1, -1), 1, BorderType.Default, new MCvScalar());
                    }

                    CvInvoke.BitwiseNot(charRect, charRect);

                    if (showSegments)
                    {
                        imageViewer.Image = charRect;
                        imageViewer.ShowDialog();
                    }

                    if (saveSegments)
                    {
                        charRect.Save("character" + i.ToString() + ".jpg");
                    }

                    output.Add(new CharSegment(charRect, rect));
                    CvInvoke.Rectangle(result, rect, new MCvScalar(0, 125, 255));
                }
            }

            if (showSegments)
            {
                imageViewer.Image = result;
                imageViewer.ShowDialog();
            }

            if (saveSegments)
            {
                result.Save("OCRcontours.jpg");
            }

            return output;
        }

        /// <summary>
        /// Method for OCR classification.
        /// </summary>
        /// <param name="charSegment">Image of input character.</param>
        /// <returns>Recognized character as a string.</returns>
        public string Classify(Mat charSegment)
        {
            var img = PixConverter.ToPix(charSegment.Bitmap);
            
            using (var page = ocrEngine.Process(img, PageSegMode.SingleWord))
            {
                string recognizedText = page.GetText();
                return recognizedText;
            }
        }

        /// <summary>
        /// Helper method for validation charcters.
        /// </summary>
        /// <param name="input">Image of input character.</param>
        /// <returns>True if character has a valid size, otherwise false.</returns>
        private bool VerifySizes(Mat input)
        {
            double aspectRatio = 45.0f / 77.0f;     // Traženi odnos širine i visine
            double aspectRatioError = 0.35f;        // Dozvoljena greška
            int minHeight = 22; //23                // Minimalna visina karaktera (px)
            int maxHeight = 28;                     // Minimalna visina karaktera (px)

            double charAspect = (double)input.Size.Width / (double)input.Size.Height;
            double minAspect = 0.10;
            double maxAspect = aspectRatio + aspectRatio * aspectRatioError;

            float area = CvInvoke.CountNonZero(input);
            float bbArea = input.Size.Width * input.Size.Height;

            // Percentage of pixels in area
            float percPixels = area / bbArea;

            if (percPixels <= 0.99 && charAspect > minAspect && charAspect < maxAspect && input.Size.Height >= minHeight && input.Size.Width < maxHeight)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        /// <summary>
        /// Helper method for normalization characters image.
        /// </summary>
        /// <param name="input">Image of input character.</param>
        /// <returns>Normalized image of character.</returns>
        private Mat PreprocessChar(Mat input)
        {
            ImageViewer viewer = new ImageViewer();
            Matrix<float> transformMat = new Matrix<float>(2, 3);
            
            int h = input.Rows;
            int w = input.Cols;
            int max = Math.Max(w, h) + 10;

            transformMat.SetIdentity();
            transformMat[0, 2] = (float)(max / 2 - w / 2);
            transformMat[1, 2] = (float)(max / 2 - h / 2 + 3);

            Mat warpImage = new Mat(max + 6, max, input.Depth, input.NumberOfChannels);
            CvInvoke.WarpAffine(input, warpImage, transformMat, warpImage.Size, Inter.Linear, Warp.Default, BorderType.Constant, new MCvScalar(0));

            return warpImage;
        }

        /// <summary>
        /// Helper method for merge contours if they enough closer.
        /// </summary>
        /// <param name="contours">List of found contours.</param>
        /// <returns>List of merged contours.</returns>
        private VectorOfVectorOfPoint MergeContours(VectorOfVectorOfPoint contours)
        {
            double maxDistanceX = 2.5;
            double maxDistanceY = 15;
            double minCharHeight = 27;
            VectorOfVectorOfPoint mergedContours = new VectorOfVectorOfPoint();

            for (int i = 0; i < contours.Size; i++)
            {
                var contour = contours[i];
                bool merged = false;
                var minRect = CvInvoke.MinAreaRect(contour);

                // Ispravljanje rotacije ukoliko postoji (nekad se vrati region sa negativnim uglom)
                var widthI = minRect.Size.Width;
                if (minRect.Angle < 0)
                {
                    widthI = minRect.Size.Height;
                }

                if (widthI < minCharHeight)
                {
                    for (int j = 0; j < mergedContours.Size; j++)
                    {
                        var mergedContour = mergedContours[j];

                        int x1 = (int)(contour.ToArray().Average(p => p.X));
                        int y1 = (int)(contour.ToArray().Average(p => p.Y));
                        int x2 = (int)(mergedContour.ToArray().Average(p => p.X));
                        int y2 = (int)(mergedContour.ToArray().Average(p => p.Y));

                        Point center1 = new Point(x1, y1);
                        Point center2 = new Point(x2, y2);
                        double distanceX = Math.Abs(center1.X - center2.X);
                        double distanceY = Math.Abs(center1.Y - center2.Y);

                        // Ako su dovoljno blizu, spajamo ih
                        if (distanceX < maxDistanceX && distanceY < maxDistanceY)
                        {
                            mergedContour.Push(contour);
                            merged = true;
                            break;
                        }
                    }

                    if (!merged)
                    {
                        mergedContours.Push(contour);
                    }
                }
                
            }
            
            return mergedContours;
        }
    }
}