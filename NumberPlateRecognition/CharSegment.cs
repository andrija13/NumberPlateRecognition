using Emgu.CV;

using System.Drawing;

namespace NumberPlateRecognition
{
    public class CharSegment
    {
        public Mat CharImage { get; set; }

        public Rectangle Position { get; set; }

        public CharSegment(Mat charImage, Rectangle position)
        {
            CharImage = charImage;
            Position = position;
        }
    }
}
