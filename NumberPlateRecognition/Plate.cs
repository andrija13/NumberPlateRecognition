using Emgu.CV;

using System.Drawing;
using System.Collections.Generic;

namespace NumberPlateRecognition
{
    public class Plate
    {
        public Mat PlateImage { get; set; }

        public Rectangle Position { get; set; }

        public List<char> Chars { get; set; } = new List<char>();

        public List<Rectangle> CharsPositions { get; set; } = new List<Rectangle>();

        public Plate(Mat plateImage, Rectangle position)
        {
            PlateImage = plateImage;
            Position = position;
        }

        public override string ToString()
        {
            var result = string.Empty;
            foreach (char c in Chars)
            {
                result += c;
            }

            return result;
        }
    }
}
