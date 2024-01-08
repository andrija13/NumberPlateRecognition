using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace NumberPlateRecognition
{
    public static class SyntaxHelper
    {
        public static Regex RegexLetters { get; set; } = new Regex(@"[A-Ž]{2}");
        public static Regex RegexNumbers { get; set; } = new Regex(@"\b\d{3,4}\b");

        public static bool MatchLetters(string letters)
        {
            var match = RegexLetters.Match(letters);

            if (match.Success)
            {
                return true;
            }

            return false;
        }

        public static bool MatchNumbers(string numbers)
        {
            var match = RegexNumbers.Match(numbers);

            if (match.Success)
            {
                return true;
            }

            return false;
        }

        public static string ChangeLetters(string input)
        {
            foreach(var c in input)
            {
                switch(c)
                {
                    case '0':
                        input = input.Replace(c, 'O');
                        break;
                    case '1':
                        input = input.Replace(c, 'Z');
                        break;
                    case '2':
                        input = input.Replace(c, 'Z');
                        break;
                    case '6':
                        input = input.Replace(c, 'G');
                        break;
                    case '8':
                        input = input.Replace(c, 'B');
                        break;
                }
            }

            return input;
        }

        public static string ChangeNumbers(string input)
        {
            foreach (var c in input)
            {
                switch (c)
                {
                    case 'D':
                    case 'H':
                    case 'O':
                    case 'U':
                        input.Replace(c, '0');
                        break;
                    case 'B':
                        input = input.Replace(c, '8');
                        break;
                    case 'I':
                        input = input.Replace(c, '1');
                        break;
                    case 'G':
                        input = input.Replace(c, '6');
                        break;
                }
            }

            return input;
        }
    }
}
