using System;
using System.IO;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace ImageRecognitionApi.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ImageRecognitionController : ControllerBase
    {

        [HttpPost]
        public IActionResult Post([FromForm]UploadedImage uploadedFile)
        {
            string imagePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "images");
            string imageName = DateTime.Now.ToString("yyyyMMddHHmmss") + uploadedFile.Image.FileName; 

            if (!Directory.Exists(imagePath))
            {
                Directory.CreateDirectory(imagePath);
            }

            try
            {
                using (FileStream fileStream = System.IO.File.Create(Path.Combine(imagePath, imageName)))
                {
                    uploadedFile.Image.CopyTo(fileStream);
                    fileStream.Flush();
                }
                var classifier = new Classifier();
                var result = classifier.Classify(imagePath, imageName);

                return File(result, "image/jpeg");
            }
            catch (Exception ex)
            {
                return Content(ex.Message.ToString());
            }
            finally
            {
                System.IO.File.Delete(Path.Combine(imagePath, imageName));
            }
        }

        public class UploadedImage
        {
            public IFormFile Image { get; set; }
        }
    }
}
