using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using Microsoft.ML;
using ObjectDetection;
using ObjectDetection.DataStructures;
using ObjectDetection.YoloParser;

namespace ImageRecognitionApi
{
    public class Classifier
    {
        public byte[] Classify(string imagePath, string imageName)
        {
            var modelFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..\\..\\..", "assets", "Model", "TinyYolo2_model.onnx");

            MLContext mlContext = new MLContext();

            try
            {
                // Load Data
                ImageNetData data = new ImageNetData()
                {
                    ImagePath = Path.Combine(imagePath, imageName),
                    Label = imageName
                };
                IEnumerable<ImageNetData> image = new List<ImageNetData>() { data };
                IDataView imageDataView = mlContext.Data.LoadFromEnumerable(image);

                // Create instance of model scorer
                var modelScorer = new OnnxModelScorer(modelFilePath, mlContext);

                // Use model to score data
                IEnumerable<float[]> probabilities = modelScorer.Score(imageDataView);

                // Post-process model output
                YoloOutputParser parser = new YoloOutputParser();

                var boundingBoxes =
                    probabilities
                    .Select(probability => parser.ParseOutputs(probability))
                    .Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F)).First();

                // Draw bounding boxes for detected objects in each of the images
                IList<YoloBoundingBox> detectedObjects = boundingBoxes;

                return DrawBoundingBox(imagePath, imageName, detectedObjects);
            }
            catch (Exception)
            {
                return null; //TODO
            }
        }

        private byte[] DrawBoundingBox(string inputImageLocation, string imageName, IList<YoloBoundingBox> filteredBoundingBoxes)
        {
            using (Image image = Image.FromFile(Path.Combine(inputImageLocation, imageName)))
            {
                var originalImageHeight = image.Height;
                var originalImageWidth = image.Width;

                foreach (var box in filteredBoundingBoxes)
                {
                    // Get Bounding Box Dimensions
                    var x = (uint)Math.Max(box.Dimensions.X, 0);
                    var y = (uint)Math.Max(box.Dimensions.Y, 0);
                    var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
                    var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);

                    // Resize To Image
                    x = (uint)originalImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth;
                    y = (uint)originalImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight;
                    width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth;
                    height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight;

                    // Bounding Box Text
                    string text = $"{box.Label}";

                    using (Graphics thumbnailGraphic = Graphics.FromImage(image))
                    {
                        thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                        thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
                        thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

                        // Define Text Options
                        Font drawFont = new Font("Arial", 12, FontStyle.Bold);
                        SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
                        SolidBrush fontBrush = new SolidBrush(Color.Black);
                        Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

                        // Define BoundingBox options
                        Pen pen = new Pen(box.BoxColor, 3.2f);
                        SolidBrush colorBrush = new SolidBrush(box.BoxColor);

                        // Draw text on image 
                        thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                        thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);

                        // Draw bounding box on image
                        thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
                    }
                }

                using (var ms = new MemoryStream())
                {
                    image.Save(ms, System.Drawing.Imaging.ImageFormat.Jpeg);
                    return ms.ToArray();
                }
            }
        }
    }
}

