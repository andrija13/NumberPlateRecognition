using Emgu.CV;
using Emgu.CV.UI;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using System;
using System.Drawing;
using System.Collections.Generic;

namespace NumberPlateRecognition
{
    public static class DetectRegion
    {
        /// <summary>
        /// Method for segmentation image of cars.
        /// </summary>
        /// <param name="inputImage">Original image.</param>
        /// <param name="showSegments">Show segments?</param>
        /// <param name="saveSegments">Save segments?</param>
        /// <returns>List of possible plates.</returns>
        public static List<Plate> ExtractSegments(Image<Bgr, byte> inputImage, bool showSegments = false, bool saveSegments = false)
        {
            var output = new List<Plate>();
            var imageViewer = new ImageViewer();

            // Grayscale
            Image<Gray, Byte> grayImage = inputImage.Convert<Gray, Byte>();
            Image<Gray, Byte> img = grayImage;
            if (saveSegments)
            {
                img.Save("grayscale.jpg");
            }

            // Gaussian blur
            CvInvoke.GaussianBlur(img, img, new Size(5, 5), 0, 0);
            if (saveSegments)
            {
                img.Save("gaussian.jpg");
            }

            // Treshold
            CvInvoke.Threshold(img, img, 0, 255, ThresholdType.Otsu | ThresholdType.Binary);
            if (saveSegments)
            {
                img.Save("treshold.jpg");
            }

            // Sobel
            CvInvoke.Sobel(img, img, DepthType.Cv8U, 1, 0); // x derivative
            if (saveSegments)
            {
                img.Save("sobel.jpg");
            }

            // Close/open morphological operation
            var structureElement = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(25, 7), new Point(-1, -1));
            CvInvoke.MorphologyEx(img, img, MorphOp.Close, structureElement, new Point(-1, -1), 2, BorderType.Default, new MCvScalar());
            if (saveSegments)
            {
                img.Save("erosion.jpg");
            }

            structureElement = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(5, 13), new Point(-1, -1));
            CvInvoke.MorphologyEx(img, img, MorphOp.Open, structureElement, new Point(-1, -1), 2, BorderType.Default, new MCvScalar());
            if (saveSegments)
            {
                img.Save("erosion1.jpg");
            }

            structureElement = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(50, 5), new Point(-1, -1));
            CvInvoke.MorphologyEx(img, img, MorphOp.Open, structureElement, new Point(-1, -1), 2, BorderType.Default, new MCvScalar());
            if (saveSegments)
            {
                img.Save("erosion2.jpg");
            }

            // Finding contours - potential regions
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            Mat hier = new Mat();
            CvInvoke.FindContours(img, contours, hier, RetrType.External, ChainApproxMethod.ChainApproxNone);

            // Validation regions
            List<RotatedRect> rects = new List<RotatedRect>();
            for (int i = 0; i < contours.Size; i++)
            {
                VectorOfPoint contour = contours[i];
                RotatedRect boundingBox = CvInvoke.MinAreaRect(contour);

                //CvInvoke.Rectangle(inputImage, boundingBox.MinAreaRect(), new MCvScalar(0, 200, 0));
                //imageViewer.Image = inputImage;
                //imageViewer.ShowDialog();

                if (VerifySizes(boundingBox))
                {
                    rects.Add(boundingBox);
                }
            }

            // Flood fill algorithm and extract region
            for (int i = 0; i < rects.Count; i++)
            {
                // Create mask for flood fill
                Mat mask = new Mat(img.Rows + 2, img.Cols + 2, DepthType.Cv8U, 1);
                mask.SetTo(new MCvScalar(0));

                Point[] seeds = GetSeeds(rects[i]);

                // Minimalna i maksimalna razlika u svetlosti/boji za flood fill
                MCvScalar lowerDiff = new MCvScalar(30, 30, 30);
                MCvScalar upperDiff = new MCvScalar(30, 30, 30);
                int flags = 4 + (255 << 8) + (int)Emgu.CV.CvEnum.FloodFillType.FixedRange + (int)Emgu.CV.CvEnum.FloodFillType.MaskOnly;

                foreach (var seed in seeds)
                {
                    CvInvoke.FloodFill(inputImage, mask, seed, new MCvScalar(255, 0, 0), out var rect, lowerDiff, upperDiff, flags: (FloodFillType)flags);
                }

                //imageViewer.Image = mask;
                //imageViewer.ShowDialog();
                //mask.Save("mask" + i + ".jpg");

                // Extract all point from mask which are white color
                List<Point> pointsInterest = new List<Point>();
                for (int y = 0; y < mask.Rows; y++)
                {
                    for (int x = 0; x < mask.Cols; x++)
                    {
                        if (mask.GetData(y, x)[0] == 255)
                        {
                            pointsInterest.Add(new Point(x, y));
                        }
                    }
                }

                PointF[] pointsInterestArray = Array.ConvertAll(pointsInterest.ToArray(), point => new PointF(point.X, point.Y));
                RotatedRect minRect = CvInvoke.MinAreaRect(pointsInterestArray);

                if (VerifySizes(minRect))
                {
                    // Ispravljanje rotacije ukoliko postoji (nekad se vrati region sa negativnim uglom)
                    var r = minRect.Size.Width / minRect.Size.Height;
                    var angle = minRect.Angle;
                    if (r < 1)
                    {
                        angle += 90;
                    }

                    var rotMat = new Mat();
                    CvInvoke.GetRotationMatrix2D(minRect.Center, angle, 1, rotMat);

                    var imgRotated = new Mat();
                    CvInvoke.WarpAffine(inputImage, imgRotated, rotMat, inputImage.Size, Inter.Cubic);

                    Size rectSize = minRect.Size.ToSize();
                    if (r < 1)
                    {
                        int temp = rectSize.Width;
                        rectSize.Width = rectSize.Height;
                        rectSize.Height = temp;
                    }

                    // Isecanje pravougaonika
                    Mat imgCrop = new Mat();
                    CvInvoke.GetRectSubPix(inputImage, rectSize, minRect.Center, imgCrop);

                    // Uskladjujemo velicinu i svetlost
                    Mat resultResized = new Mat(33, 144, DepthType.Cv8U, 3);
                    CvInvoke.Resize(imgCrop, resultResized, resultResized.Size, 0, 0, Inter.Linear);

                    // Normalizacija
                    Mat grayResult = new Mat();
                    CvInvoke.CvtColor(resultResized, grayResult, ColorConversion.Bgr2Gray);
                    CvInvoke.GaussianBlur(grayResult, grayResult, new Size(3, 3), 0, 0);

                    // Izjednačavanje histograma slike (osvetljenje i kontrast)
                    CvInvoke.EqualizeHist(grayResult, grayResult);

                    if (showSegments)
                    {
                        imageViewer.Image = grayResult;
                        imageViewer.ShowDialog();
                    }

                    if (saveSegments)
                    {
                        grayResult.Save("plate" + i.ToString() + ".jpg");
                    }

                    output.Add(new Plate(grayResult, minRect.MinAreaRect()));
                }
            }

            return output;
        }

        /// <summary>
        /// Helper method for validation regions.
        /// </summary>
        /// <param name="boundingBox">Image of potential region.</param>
        /// <returns>True if region has a valid size, otherwise false.</returns>
        private static bool VerifySizes(RotatedRect boundingBox)
        {
            double aspectRatioThreshold = 5;         // Traženi odnos širine i visine
            double aspectRatioErrorMargin = 0.6;     // Dozvoljena greška
            int minPlateHeight = 15;                 // Minimalna visina tablice (px)
            int maxPlateHeight = 250;                // Maksimalna visina tablice (px)

            double aspectRatio = boundingBox.Size.Width / boundingBox.Size.Height;
            int regionHeight = (int)boundingBox.Size.Height;

            if (aspectRatio < 1)
            {
                aspectRatio = boundingBox.Size.Height / boundingBox.Size.Width;
                regionHeight = (int)boundingBox.Size.Width;
            }

            // Provera odnosa širine i visine uzimajući u obzir dozvoljenu grešku
            if (Math.Abs(aspectRatio - aspectRatioThreshold) <= aspectRatioThreshold * aspectRatioErrorMargin)
            {
                // Provera visine regiona
                if (regionHeight >= minPlateHeight && regionHeight <= maxPlateHeight)
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Helper method for generate random seed points for Flood Fill algorithm.
        /// </summary>
        /// <param name="boundingBox"> Image of potential region.</param>
        /// <returns>Array of generated seeds.</returns>
        private static Point[] GetSeeds(RotatedRect boundingBox)
        {
            PointF center = boundingBox.Center;
            float minSize = (boundingBox.Size.Width < boundingBox.Size.Height) ? boundingBox.Size.Width : boundingBox.Size.Height;
            minSize = minSize - minSize * 0.5f;

            // Generisanje nasumicnih seed tačaka u blizini centra
            int numSeeds = 10;
            Point[] seeds = new Point[numSeeds];
            Random random = new Random();
            for (int i = 0; i < numSeeds; i++)
            {
                float xOffset = random.Next((int)minSize) - (minSize / 2);
                float yOffset = random.Next((int)minSize) - (minSize / 2);
                seeds[i] = new Point((int)(center.X + xOffset), (int)(center.Y + yOffset));
            }

            return seeds;
        }
    }
}
