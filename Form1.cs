using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace CarNumberRecognizer
{
    public partial class Form1 : Form
    {
        // распознование номеров машин
        private NumberPlateRecognizer plateRecognizer;

        private Point startPoint;

        private Mat inputImage;

        public Form1()
        {
            InitializeComponent();
        }

        private void ProcessImage(IInputOutputArray image)
        {
            List<IInputOutputArray> licensePlateImageList = new List<IInputOutputArray>();
            List<IInputOutputArray> filteredLicensePlatImageList = new List<IInputOutputArray>();
            List<RotatedRect> licenseBoxList = new List<RotatedRect>();

            List<string> recognizedPlates = plateRecognizer.DetectLicensePlates(image,
                licensePlateImageList, filteredLicensePlatImageList,
                licenseBoxList);

            panel1.Controls.Clear();

            startPoint = new Point(10, 10);

            for (int i = 0; i < recognizedPlates.Count; i++)
            {
                Mat dest = new Mat();

                CvInvoke.VConcat(licensePlateImageList[i], filteredLicensePlatImageList[i], dest);

                AddLabelAndImage($"Номер: {recognizedPlates[i]}", dest);
            }

            Image<Bgr, byte> outputImage = inputImage.ToImage<Bgr, byte>();

            foreach (RotatedRect rect in licenseBoxList)
            {
                PointF[] v = rect.GetVertices();

                PointF prevPoint = v[0];
                PointF firstPoint = prevPoint;
                PointF nextpoint = prevPoint;
                PointF lastpoint = nextpoint;

                for (int i = 1; i < v.Length; i++)
                {
                    nextpoint = v[i];

                    CvInvoke.Line(outputImage, Point.Round(prevPoint), Point.Round(nextpoint), new MCvScalar(0, 0, 255), 5,
                        LineType.EightConnected, 0);

                    prevPoint = nextpoint;
                    lastpoint = prevPoint;
                }

                CvInvoke.Line(outputImage, Point.Round(lastpoint), Point.Round(firstPoint), new MCvScalar(0, 0, 255), 5,
                    LineType.EightConnected, 0);
            }

            pictureBox1.Image = outputImage.Bitmap;
        }

        private void AddLabelAndImage(string labelText, IInputArray image)
        {
            Label label = new Label();

            label.Text = labelText;
            label.Width = 100;
            label.Height = 30;
            label.Location = startPoint;

            startPoint.Y += label.Height;

            panel1.Controls.Add(label);

            PictureBox box = new PictureBox();

            Mat m = image.GetInputArray().GetMat();

            box.ClientSize = m.Size;
            box.Image = m.Bitmap;
            box.Location = startPoint;

            startPoint.Y += box.Height + 10;

            panel1.Controls.Add(box);
        }

        private void открытьToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                if (openFileDialog1.ShowDialog() == DialogResult.OK)
                {
                    pictureBox1.Image = Image.FromFile(openFileDialog1.FileName);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Ошибка!", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            plateRecognizer = new NumberPlateRecognizer(@"c:\Prog\", "rus");
        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            inputImage = new Mat(openFileDialog1.FileName);

            UMat um = inputImage.GetUMat(AccessType.ReadWrite);

            ProcessImage(um);
        }
    }
}
