using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenCvSharp;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

using System.IO;


namespace errerRect
{
    class Program
    {
        struct ErrorRect
        {
           public int iIdx;
           public Rect Rrect;
           public Rect RoriginRect;

            public ErrorRect(int x, Rect r)
            {
                iIdx = x;
                Rrect = r;
                RoriginRect = r;
            }
        };
        public enum eDir
        {
            TOP,
            BOTTOM,
            LEFT,
            RIGHT,
        }

        static private void Fourrulecalculations(Mat x, Mat y) //연습코드
        {
            Mat add = new Mat();
            Mat sub = new Mat();
            Mat mul = new Mat();
            Mat div = new Mat();
            Mat max = new Mat();
            Mat min = new Mat();

            Cv2.Add(x, y, add);
            Cv2.Subtract(x, y, sub);
            Cv2.Multiply(x, y, mul);
            Cv2.Divide(x, y, div);
            Cv2.Max(x, y, max);
            Cv2.Min(x, y, min);

            Cv2.Resize(x, x, new Size(400, 300));
            Cv2.Resize(y, y, new Size(400, 300));
            Cv2.Resize(add, add, new Size(400, 300));
            Cv2.Resize(sub, sub, new Size(400, 300));
            Cv2.Resize(mul, mul, new Size(400, 300));
            Cv2.Resize(div, div, new Size(400, 300));
            Cv2.Resize(max, max, new Size(400, 300));
            Cv2.Resize(min, min, new Size(400, 300));


            Cv2.ImShow("origin1", x);
            Cv2.ImShow("origin2", y);
            Cv2.ImShow("add", add);
            Cv2.ImShow("sub", sub);
            Cv2.ImShow("mul", mul);
            Cv2.ImShow("div", div);
            Cv2.ImShow("max", min);
            Cv2.ImShow("min", max);

            Cv2.WaitKey(0);
        }

        static private void Blur(Mat x) //연습코드
        {
            Mat blur = new Mat();
            Mat box_filter = new Mat();
            Mat median_blur = new Mat();
            Mat gaussian_blur = new Mat();
            Mat bilateral_filter = new Mat();

            Cv2.Blur(x, blur, new Size(9, 9), new Point(-1, -1), BorderTypes.Default);
            Cv2.BoxFilter(x, box_filter, MatType.CV_8UC3, new Size(9, 9), new Point(-1, -1), true, BorderTypes.Default);
            Cv2.MedianBlur(x, median_blur, 9);
            Cv2.GaussianBlur(x, gaussian_blur, new Size(9, 9), 1, 1, BorderTypes.Default);
            Cv2.BilateralFilter(x, bilateral_filter, 9, 3, 3, BorderTypes.Default);

            Cv2.Resize(x, x, new Size(400, 300));
            Cv2.Resize(blur, blur, new Size(400, 300));
            Cv2.Resize(box_filter, box_filter, new Size(400, 300));
            Cv2.Resize(median_blur, median_blur, new Size(400, 300));
            Cv2.Resize(gaussian_blur, gaussian_blur, new Size(400, 300));
            Cv2.Resize(bilateral_filter, bilateral_filter, new Size(400, 300));

            Cv2.ImShow("orign", x);
            Cv2.ImShow("blur", blur);
            Cv2.ImShow("box_filter", box_filter);
            Cv2.ImShow("median_blur", median_blur);
            Cv2.ImShow("gaussian_blur", gaussian_blur);
            Cv2.ImShow("bilateral_filter", bilateral_filter);
            Cv2.WaitKey(0);
        }

        static private unsafe void ConvolutionBlur(ref Mat _image) //연습코드
        {
            Cv2.Resize(_image, _image, new Size(800, 600));
            //Cv2.CvtColor(_image, _image, ColorConversionCodes.RGB2GRAY);
            //Cv2.ImShow("orign", _image);

            float[] fSampleMask = new float[] {0.1f, 0.1f, 0.1f
                                            , 0.1f, 0.1f, 0.1f
                                            , 0.1f, 0.1f, 0.1f };

            //픽셀순회
            for (int i = 1; i < _image.Cols - 1; ++i)
            {
                for (int j = 1; j < _image.Rows - 1; ++j)
                {
                    Vec3b ResultPixel = new Vec3b();
                    double[] fResult = new double[3] { 0.0f, 0.0f, 0.0f };

                    //마스크 순회
                    for (int y = 0; y < 3; ++y)
                    {
                        for (int x = 0; x < 3; ++x)
                        {
                            var tempPixel1 = _image.At<Vec3b>(j + x - 1, i + y - 1).Item0 * (fSampleMask[y * 3 + x]);
                            var tempPixel2 = _image.At<Vec3b>(j + x - 1, i + y - 1).Item1 * (fSampleMask[y * 3 + x]);
                            var tempPixel3 = _image.At<Vec3b>(j + x - 1, i + y - 1).Item2 * (fSampleMask[y * 3 + x]);
                            fResult[0] += tempPixel1;
                            fResult[1] += tempPixel2;
                            fResult[2] += tempPixel3;
                        }
                    }

                    //오버플로우 대비
                    for (int l = 0; l < 3; ++l)
                    {
                        if (fResult[l] < 0)
                            fResult[l] = 0;

                        if (fResult[l] > 255)
                            fResult[l] = 255;
                        
                        ResultPixel[l] = (byte)fResult[l];
                    }

                    _image.Set<Vec3b>(j, i, ResultPixel);
                }
            }
            Cv2.ImShow("blur", _image);
            //ConvolutionSharp(_image);
            //Cv2.WaitKey(0);
        }

        static private unsafe void ConvolutionSharp(Mat _image) //연습코드
        {
            //Cv2.Resize(_image, _image, new Size(800, 600));
            //Cv2.CvtColor(_image, _image, ColorConversionCodes.RGB2GRAY);

            //Cv2.ImShow("orign", _image);
            Mat Clone = _image.Clone();
            float[] fSampleMask = new float[] {-1f, -1f, -1f
                                            ,   -1f,  9f,  -1f
                                            ,  -1f, -1f, -1f };

            InputArray inarray = InputArray.Create(fSampleMask);
            Cv2.Filter2D(Clone, Clone, Clone.Type(), inarray, new Point(0, 0));
            Cv2.ImShow("blur to Func Sharp", Clone);

            //픽셀순회
            for (int i = 1; i < _image.Cols - 1; i+=2)
            {
                for (int j = 1; j < _image.Rows - 1; j+=2)
                {
                    Vec3b ResultPixel = new Vec3b();
                    double[] fResult = new double[3] { 0.0f, 0.0f, 0.0f };

                    //마스크 순회
                    for (int y = 0; y < 3; ++y)
                    {
                        for (int x = 0; x < 3; ++x)
                        {
                            var tempPixel1 = _image.At<Vec3b>(j + x - 1, i + y - 1).Item0 * (fSampleMask[y * 3 + x]);
                            var tempPixel2 = _image.At<Vec3b>(j + x - 1, i + y - 1).Item1 * (fSampleMask[y * 3 + x]);
                            var tempPixel3 = _image.At<Vec3b>(j + x - 1, i + y - 1).Item2 * (fSampleMask[y * 3 + x]);
                            fResult[0] += tempPixel1;
                            fResult[1] += tempPixel2;
                            fResult[2] += tempPixel3;
                        }
                    }

                    //오버플로우시 최대값 최소값 변경
                    for (int l = 0; l < 3; ++l)
                    {
                        if (fResult[l] < 0)
                            fResult[l] = 0;

                        if (fResult[l] > 255)
                            fResult[l] = 255;

                        ResultPixel[l] = (byte)fResult[l];
                    }

                    _image.Set<Vec3b>(j, i, ResultPixel);
                }
            }
            Cv2.ImShow("mySharp", _image);

            Cv2.WaitKey(0);
        }

        static private unsafe void ConvolutionEdge(Mat _image, ref List<List<Point>> p ) //연습코드
        {
            ConvolutionBlur(ref _image);

            Mat Temp = _image.Clone();

            // 마스크
            float[] fSampleMask3_1 = new float[] {-1f, -2f, -1f
                                            ,    0f,    0f,  0f
                                            ,    1f,    2f,  1f};

            float[] fSampleMask3_2 = new float[] {1f,  0f,  -1f
                                            ,     2f,  0f,  -2f
                                            ,     1f,  0f,  -1f};

            List<Point> lXY = new List<Point>();

            //픽셀순회 좌에서 우로
            for (int i = 1; i < _image.Rows - 1; i+=1)
            {
                for (int j = 1; j < _image.Cols - 1; j+=1)
                {
                    Byte ResultPixel = new Byte();
                    double fResult  = 0;
                    double fResult2 = 0;
                    double fResult3 = 0;

                    //마스크 순회
                    for (int y = 0; y < 3; ++y)
                    {
                        for (int x = 0; x < 3; ++x)
                        {
                            var tempPixel1 = Temp.At<byte>(i + y - 1, j + x - 1) * (fSampleMask3_1[y * 3 + x]);
                            fResult += tempPixel1;

                            tempPixel1 = Temp.At<byte>(i + y - 1, j + x - 1) * (fSampleMask3_2[y * 3 + x]);
                            fResult2 += tempPixel1;
                        }
                    }

                    //오버플로우 변환
                    if (fResult < 0)
                        fResult *= -1f;

                    if (fResult > 255)
                        fResult = 255f;

                    if (fResult2 < 0)
                        fResult2 *= -1f;

                    if (fResult2 > 255)
                        fResult2 = 255f;

                    fResult = fResult * fResult;
                    fResult2 = fResult2 * fResult2;
                    fResult3 = fResult + fResult2;
                    

                    ResultPixel = (byte)fResult3;

                    if (ResultPixel > 100)
                    {
                        ResultPixel = 255;
                        lXY.Add(new Point(j, i));
                    }

                    _image.Set<Byte>(i, j, ResultPixel);

                    //좌표검출
                    //Top position
                    if (j > 19 && i > 43 && j < 638 && i < 85)
                    {
                        if (_image.At<Byte>(i, j) == 255)
                            p[(int)eDir.TOP].Add(new Point(j, i));
                    }
                    //bottom position
                    if (j > 28 && i > 381 && j < 664 && i < 520)
                    {
                        if (_image.At<Byte>(i, j) == 255)
                            p[(int)eDir.BOTTOM].Add(new Point(j, i));
                    }
                    //Left position
                    if (j > 19 && i > 43 && j < 64 && i < 524)
                    {
                        if (_image.At<Byte>(i, j) == 255)
                            p[(int)eDir.LEFT].Add(new Point(j, i));
                    }
                    //Right position
                    if (j > 580 && i > 26 && j < 650 && i < 513)
                    {
                        if (_image.At<Byte>(i, j) == 255)
                            p[(int)eDir.RIGHT].Add(new Point(j, i));
                    }
                }
            }
            Cv2.ImShow("findEdge", _image);
        }
        
        static void FindLine(ref List<Point> _lPosition, Random _rm, ref double _tanSata, ref double _y_inter)
       {
            while (true)
            {
                int Iidx; int Iidx2;
                Iidx = _rm.Next(0, _lPosition.Count());
                Iidx2 = _rm.Next(0, _lPosition.Count());

                //직선의 방정식 y = ax +b
                int x1 = _lPosition[Iidx].X;
                int y1 = _lPosition[Iidx].Y;
                int x2 = _lPosition[Iidx2].X;
                int y2 = _lPosition[Iidx2].Y;

                if (x2 == x1)
                    continue;
                double a = (double)(y2 - y1) / (double)(x2 - x1); // 기울기
                double b = (a * x1) * -1 + y1;    // y절편
                double result = 0;
                for (int i = 0; i < _lPosition.Count(); i++)
                {
                    double distance = _lPosition[i].Y - b - (a * _lPosition[i].X) / Math.Sqrt((a) * (a) + 1);
                    if (distance < 0)
                        distance *= -1;
                    result += distance;
                }
                result /= _lPosition.Count();
                if (result < 100)
                {
                    _tanSata = a;
                    _y_inter = b;
                    break;
                }
            }
        }

        static private void ChangeColor(int x, ref Scalar c)
        {
            switch (x)
            {
                case 0:
                    c = Scalar.Red;
                    break;
                case 1:
                    c = Scalar.Orange;
                    break;
                case 2:
                    c = Scalar.Yellow;
                    break;
                case 3:
                    c = Scalar.Green;
                    break;
                case 4:
                    c = Scalar.Blue;
                    break;
                case 5:
                    c = Scalar.Green;
                    
                    break;
                case 6:
                    c = Scalar.Purple;
                    break;
            }
        }
        static int Max(int x, int y)
        {
            if (x > y)
                return x;

            return y;
        }
        static int Min(int x, int y)
        {
            if (x < y)
                return x;

            return y;
        }


        static void Main(string[] args)
        {
           //string ppath =  "D:\\Projects\\EditPhotometricStereo\\Debug\\image\\chrome4\\chrome4.0.bmp";
           //Mat iimage = Cv2.ImRead(ppath, 0);
           //byte aaa = iimage.At<byte>(0, 2);


            string str = System.IO.Directory.GetCurrentDirectory();
            str += "\\Image\\";
            System.IO.DirectoryInfo di = new System.IO.DirectoryInfo(str);
            Mat lBinaryMat = new Mat();
            List<List<Mat>> listImage = new List<List<Mat>>();
            List<List<Mat>> listCutImage = new List<List<Mat>>();
            List<string> sFilePropertyList = new List<string>();

            int num = 0;

            //foreach (var item in di.GetFiles())
            //{
            //    string strpath = str + item.Name;
            //    string strtext = str + String.Format("{0}.txt", num);
            //    //fs = new FlieStream(filepath, FileMode.Create, FileAccess.Write);
            //    FileStream fs = File.Create(strtext);//new FileStream(strtext, FileMode.Truncate, FileAccess.ReadWrite);
            //    BinaryWriter sw = new BinaryWriter(fs);

            //    lBinaryMat = (Cv2.ImRead(strpath, ImreadModes.Grayscale));
            //    for (int yy = 0; yy < lBinaryMat.Rows; ++yy)
            //    {
            //        for (int xx = 0; xx < lBinaryMat.Cols; ++xx)
            //        {
            //            Byte b = lBinaryMat.At<byte>(yy, xx);
            //            sw.Write(b);
            //        }
            //    }
            //    ++num;
            //    sw.Close();
            //    fs.Close();
            //}


            int iRefIdx = 3;
            int listnum = -1;
            //해당 폴더에 있는 파일이름가져오기와서 묶음으로 묶기
            foreach (var item in di.GetFiles())
            {
                if (item.Name.IndexOf("IN2") != -1)
                {
                    listImage.Add(new List<Mat>());
                    listCutImage.Add(new List<Mat>());
                    ++listnum;
                }

                string strpath = str + item.Name;
                listImage[listnum].Add(Cv2.ImRead(strpath, ImreadModes.Unchanged));
                listCutImage[listnum].Add(Cv2.ImRead(strpath, ImreadModes.Color));
                //묶음의 끝에 도달하면 바로 불량 검사를진행한다.
                if (item.Name.IndexOf("outring") != -1)
                {
                    //convolution 연습
                    //ConvolutionBlur(listCutImage[listnum][iRefIdx]);
                    //ConvolutionSharp(listCutImage[listnum][iRefIdx]);

                    List<List<Point>> lPosition = new List<List<Point>>();

                    for(int c =0; c< 4; ++c)
                        lPosition.Add(new List<Point>());

                    Cv2.CvtColor(listCutImage[listnum][iRefIdx], listCutImage[listnum][iRefIdx], ColorConversionCodes.RGB2GRAY);

                    ConvolutionEdge(listCutImage[listnum][iRefIdx], ref lPosition);

                    Mat cloneCutImage = new Mat();
                    Cv2.CvtColor(listCutImage[listnum][iRefIdx], listCutImage[listnum][iRefIdx], ColorConversionCodes.GRAY2RGB);
                    cloneCutImage = listCutImage[listnum][iRefIdx].Clone();

                    Rect rc1 = Cv2.BoundingRect(lPosition[0]);
                    Rect rc2 = Cv2.BoundingRect(lPosition[1]);
                    Rect rc3 = Cv2.BoundingRect(lPosition[2]);
                    Rect rc4 = Cv2.BoundingRect(lPosition[3]);


                    Cv2.Rectangle(listCutImage[listnum][iRefIdx], rc1, Scalar.Red, 5);
                    Cv2.Rectangle(listCutImage[listnum][iRefIdx], rc2, Scalar.Red, 5);
                    Cv2.Rectangle(listCutImage[listnum][iRefIdx], rc3, Scalar.Red, 5);
                    Cv2.Rectangle(listCutImage[listnum][iRefIdx], rc4, Scalar.Red, 5);

                    Cv2.ImShow("position", listCutImage[listnum][iRefIdx]);

                    List<double> dResult = new List<double>();
                    Random rm = new Random();
                    int Iidx;
                    int Iidx2;
                    double tanSata = 0;
                    double y_intercept = 0;

                    while (true)
                    {

                        Iidx = rm.Next(0, lPosition[(int)eDir.TOP].Count());
                        Iidx2 = rm.Next(0, lPosition[(int)eDir.TOP].Count());

                        //직선의 방정식 y = ax +b
                        int x1 = lPosition[(int)eDir.TOP][Iidx].X;
                        int y1 = lPosition[(int)eDir.TOP][Iidx].Y;
                        int x2 = lPosition[(int)eDir.TOP][Iidx2].X;
                        int y2 = lPosition[(int)eDir.TOP][Iidx2].Y;

                        if (x2 == x1)
                            continue;
                        double a = (double)(y2 - y1) / (double)(x2 - x1); // 기울기
                        double b = (a * x1) * -1 + y1;    // y절편
                        double result = 0;
                        for (int i = 0; i < lPosition[(int)eDir.TOP].Count(); i++)
                        {
                            double distance = lPosition[(int)eDir.TOP][i].Y - b - (a * lPosition[(int)eDir.TOP][i].X) / Math.Sqrt((a) * (a) + 1);
                            if (distance < 0)
                                distance *= -1;
                            result += distance;
                        }
                        result /= lPosition[(int)eDir.TOP].Count();
                        if (result < 6)
                        {
                            dResult.Add(result);
                            tanSata = a;
                            y_intercept = b;
                            break;
                        }
                    }
                    

                    int dXpos1 = 40;
                    int dYpos1 = (int)(tanSata * dXpos1 + y_intercept);

                    int dXpos2 = 590;
                    int dYpos2 = (int)(tanSata * dXpos2 + y_intercept);

                    Cv2.Line(cloneCutImage, dXpos1, dYpos1, dXpos2, dYpos2, Scalar.Blue, 5);
                    

                    while (true)
                    {
                        Iidx = rm.Next(0, lPosition[(int)eDir.RIGHT].Count());
                        Iidx2 = rm.Next(0, lPosition[(int)eDir.RIGHT].Count());

                        //직선의 방정식 y = ax +b
                        int x1 = lPosition[(int)eDir.RIGHT][Iidx].X;
                        int y1 = lPosition[(int)eDir.RIGHT][Iidx].Y;
                        int x2 = lPosition[(int)eDir.RIGHT][Iidx2].X;
                        int y2 = lPosition[(int)eDir.RIGHT][Iidx2].Y;

                        if (x2 == x1)
                            continue;
                        double a = (double)(y2 - y1) / (double)(x2 - x1); // 기울기
                        double b = (a * x1) * -1 + y1;    // y절편
                        double result = 0;
                        for (int i = 0; i < lPosition[(int)eDir.RIGHT].Count(); i++)
                        {
                            double distance = (float)(lPosition[(int)eDir.RIGHT][i].Y - b - (a * lPosition[(int)eDir.RIGHT][i].X)) / (float)(Math.Sqrt((a) * (a) + 1));
                            if (distance < 0)
                                distance *= -1;
                            result += distance;
                        }
                        result /= lPosition[(int)eDir.RIGHT].Count();
                        if (result < 6)
                        {
                            dResult.Add(result);
                            tanSata = a;
                            y_intercept = b;
                            break;
                        }
                    }

                    int dYpos3 = 40;
                    int dXpos3 = (int)((dYpos3 / tanSata) - (y_intercept / tanSata));

                    int dYpos4 = 430;
                    int dXpos4 = (int)((dYpos4 / tanSata) - (y_intercept / tanSata));

                    Cv2.Line(cloneCutImage, dXpos3, dYpos3, dXpos4, dYpos4, Scalar.Blue, 5);

                    while (true)
                    {
                        Iidx = rm.Next(0, lPosition[(int)eDir.BOTTOM].Count());
                        Iidx2 = rm.Next(0, lPosition[(int)eDir.BOTTOM].Count());

                        //직선의 방정식 y = ax +b
                        int x1 = lPosition[(int)eDir.BOTTOM][Iidx].X;
                        int y1 = lPosition[(int)eDir.BOTTOM][Iidx].Y;
                        int x2 = lPosition[(int)eDir.BOTTOM][Iidx2].X;
                        int y2 = lPosition[(int)eDir.BOTTOM][Iidx2].Y;

                        if (x2 == x1)
                            continue;
                        double a = (double)(y2 - y1) / (double)(x2 - x1); // 기울기
                        double b = (a * x1) * -1 + y1;    // y절편
                        double result = 0;
                        for (int i = 0; i < lPosition[(int)eDir.BOTTOM].Count(); i++)
                        {
                            double distance = (float)(lPosition[(int)eDir.BOTTOM][i].Y - b - (a * lPosition[(int)eDir.BOTTOM][i].X)) / (float)(Math.Sqrt((a) * (a) + 1));
                            if (distance < 0)
                                distance *= -1;
                            result += distance;
                        }
                        result /= lPosition[(int)eDir.BOTTOM].Count();
                        if (result < 11.2)
                        {
                            dResult.Add(result);
                            tanSata = a;
                            y_intercept = b;
                            break;
                        }
                    }

                    dXpos1 = 40;
                    dYpos1 = (int)(tanSata * dXpos1 + y_intercept);

                    dXpos2 = 610;
                    dYpos2 = (int)(tanSata * dXpos2 + y_intercept);

                    Cv2.Line(cloneCutImage, dXpos1, dYpos1, dXpos2, dYpos2, Scalar.Blue, 5);

                    while (true)
                    {
                        Iidx = rm.Next(0, lPosition[(int)eDir.LEFT].Count());
                        Iidx2 = rm.Next(0, lPosition[(int)eDir.LEFT].Count());

                        //직선의 방정식 y = ax +b
                        int x1 = lPosition[(int)eDir.LEFT][Iidx].X;
                        int y1 = lPosition[(int)eDir.LEFT][Iidx].Y;
                        int x2 = lPosition[(int)eDir.LEFT][Iidx2].X;
                        int y2 = lPosition[(int)eDir.LEFT][Iidx2].Y;

                        if (x2 == x1)
                            continue;
                        double a = (double)(y2 - y1) / (double)(x2 - x1); // 기울기
                        double b = (a * x1) * -1 + y1;    // y절편
                        double result = 0;
                        for (int i = 0; i < lPosition[(int)eDir.LEFT].Count(); i++)
                        {
                            double distance = (float)(lPosition[(int)eDir.LEFT][i].Y - b - (a * lPosition[(int)eDir.LEFT][i].X)) / (float)(Math.Sqrt((a) * (a) + 1));
                            if (distance < 0)
                                distance *= -1;
                            result += distance;
                        }
                        result /= lPosition[(int)eDir.LEFT].Count();
                        if (result < 6)
                        {
                            dResult.Add(result);
                            tanSata = a;
                            y_intercept = b;
                            break;
                        }
                    }

                    dYpos3 = 60;
                    dXpos3 = (int)((dYpos3 / tanSata) - (y_intercept / tanSata));

                    dYpos4 = 450;
                    dXpos4 = (int)((dYpos4 / tanSata) - (y_intercept / tanSata));

                    Cv2.Line(cloneCutImage, dXpos3, dYpos3, dXpos4, dYpos4, Scalar.Blue, 5);

                    Cv2.ImShow("Line", cloneCutImage);
                    Cv2.WaitKey(0);

                    //관심 영역 추출하기
                    //이진화 시키기
                    Mat RIOImage = new Mat();
                    Mat binary = new Mat();
                    RIOImage = listImage[listnum][iRefIdx].Clone();
                    Cv2.Threshold(RIOImage, binary, 55, 255, ThresholdTypes.Binary);
                                      
                    //morphology 변환 팽창,수축(dilate) (erode)
                    Mat dilate = new Mat();
                    Mat dilate2 = new Mat();
                    Mat erode = new Mat();
                    Mat element = Cv2.GetStructuringElement(MorphShapes.Cross, new Size(5, 5));
                    Cv2.Dilate(binary, dilate, element, new Point(2, 2), 20);
                    Cv2.Erode(dilate, erode, element, new Point(-1, -1), 40);
                    Cv2.Dilate(erode, dilate2, element, new Point(2, 2), 20);

                    //contours 윤곽선 찾기
                    Point[][] contours;
                    HierarchyIndex[] hierarchy;

                    Cv2.FindContours(dilate2, out contours, out hierarchy, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);
                    Rect boundingRectSlice = new Rect();
                    Rect boundingRect = new Rect();
                    boundingRectSlice = Cv2.BoundingRect(contours[0]);
                    //Cv2.Rectangle(Mtemp[i], errTemp.RoriginRect, sVariableColor, 5);
                    Cv2.CvtColor(listImage[listnum][iRefIdx], listImage[listnum][iRefIdx], ColorConversionCodes.RGB2RGBA);
                    Cv2.Rectangle(listImage[listnum][iRefIdx], boundingRectSlice, Scalar.Red, 5);
                    //관심영역만 자르기 
                    for (int i = 0; i < listImage[listnum].Count(); ++i)
                    {
                        listCutImage[listnum][i] = listImage[listnum][i].SubMat(boundingRectSlice);
                    }
                    
                    //불량검출하기
                    //빼기 연산후 이진화 
                    //예시 Cv2.Subtract(x, y, sub);
                    Mat[] Mtemp = new Mat[7];
                    List<ErrorRect> RectList = new List<ErrorRect>();
                    element = Cv2.GetStructuringElement(MorphShapes.Cross, new Size(3, 3));
                    for (int i = 0; i < 7; ++i)
                    {
                        if (i == iRefIdx)
                            continue;
                        switch (i)
                        {
                            case 0:
                                Cv2.Threshold(listCutImage[listnum][i], binary, 15, 255, ThresholdTypes.Binary);
                                Cv2.MedianBlur(binary, binary, 9);                               
                                Cv2.Erode(binary, binary, element, new Point(-1, -1), 20);
                                Mtemp[i] = binary.Clone();
                                break;
                            case 1:
                                Cv2.Threshold(listCutImage[listnum][i], binary, 147, 255, ThresholdTypes.Binary);
                                Mtemp[i] = binary.Clone();
                                break;
                            case 2:
                                Cv2.Threshold(listCutImage[listnum][i], binary, 115, 255, ThresholdTypes.Binary);
                                Cv2.Erode(binary, binary, element, new Point(-1, -1), 3);
                                Cv2.MedianBlur(binary, binary, 9);
                                Mtemp[i] = binary.Clone();
                                break;
                            case 4:
                                Cv2.Threshold(listCutImage[listnum][i], binary, 108, 255, ThresholdTypes.Binary);
                                Cv2.Dilate(binary, binary, element, new Point(2, 2), 10);
                                Cv2.Erode(binary, binary, element, new Point(-1, -1), 10);
                                Mtemp[i] = binary.Clone();                            
                                break;
                            case 5:
                                Cv2.Threshold(listCutImage[listnum][i], binary, 73, 255, ThresholdTypes.Binary);
                                Mtemp[i] = binary.Clone();
                                break;
                            case 6:
                                Cv2.Threshold(listCutImage[listnum][i], binary, 49, 255, ThresholdTypes.Binary);
                                Cv2.Erode(binary, binary, element, new Point(-1, -1), 10);
                                Mtemp[i] = binary.Clone();
                                break;
                        }
                        

                        //윤곽선 찾기
                        Cv2.FindContours(binary, out contours, out hierarchy, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);
                        //그레이 속성을 컬러로 변경(색상있는 도형을 그리기 위해서)
                        Cv2.CvtColor(listImage[listnum][iRefIdx], listImage[listnum][iRefIdx], ColorConversionCodes.RGB2RGBA);
                        Cv2.CvtColor(Mtemp[i], Mtemp[i], ColorConversionCodes.RGB2RGBA);
                        Scalar sVariableColor = Scalar.Red;
                        foreach (Point[] p in contours)
                        {
                            double length = Cv2.ArcLength(p, true);
                            double area = Cv2.ContourArea(p, true);
                            if (length < 100 || area < 4000 || p.Length < 5) continue;
                            if (area > 80000) continue;
                            ErrorRect errTemp = new ErrorRect();
                            boundingRect = Cv2.BoundingRect(p);
                            errTemp.RoriginRect = boundingRect;
                            boundingRect.Top = boundingRect.Top + boundingRectSlice.Top;
                            boundingRect.Left = boundingRect.Left + boundingRectSlice.Left;
                            ChangeColor(i, ref sVariableColor);
                            errTemp.iIdx = i;
                            errTemp.Rrect = boundingRect;
                            RectList.Add(errTemp);

                            Cv2.Rectangle(Mtemp[i], errTemp.RoriginRect, sVariableColor, 5);
                        }
                        String strImage = String.Format("image {0}", i+1);
                        Cv2.Rectangle(Mtemp[i], RectList[i].RoriginRect, sVariableColor, 5);
                        Cv2.Resize(Mtemp[i], Mtemp[i], new Size(400, 300));
                        Cv2.ImShow(strImage, Mtemp[i]);
                    }

                    
                    
                    // 사각형중 겹치는 사각형 제거
                    for (int i = 0; i < RectList.Count(); ++i)
                    {
                        for(int j = 0; j < RectList.Count(); ++j)
                        {
                            //충돌체크
                            if(RectList[i].Rrect.Right >= RectList[j].Rrect.Left &&
                                RectList[j].Rrect.Right >= RectList[i].Rrect.Left &&
                                RectList[i].Rrect.Top <= RectList[j].Rrect.Bottom &&
                                RectList[j].Rrect.Top <= RectList[i].Rrect.Bottom)
                            {
                                //얼만큼 겹치는가 70%이상이면 지우기
                                //겹치는 부분의 넓이를 비교한다.
                                int iIarea = (RectList[i].Rrect.Right - RectList[i].Rrect.Left)
                                 * (RectList[i].Rrect.Bottom - RectList[i].Rrect.Top);

                                int iJarea = (RectList[j].Rrect.Right - RectList[j].Rrect.Left)
                                 * (RectList[j].Rrect.Bottom - RectList[j].Rrect.Top);

                                //left top right bottom
                                int x1 = RectList[i].Rrect.Left;
                                int y1 = RectList[i].Rrect.Top;
                                int x2 = RectList[i].Rrect.Right;
                                int y2 = RectList[i].Rrect.Bottom;

                                int x3 = RectList[j].Rrect.Left;
                                int y3 = RectList[j].Rrect.Top;
                                int x4 = RectList[j].Rrect.Right;
                                int y4 = RectList[j].Rrect.Bottom;

                                int left_up_x = Max(x1, x3);
                                int left_up_y = Max(y1, y3);
                                int right_down_x = Min(x2, x4);
                                int right_down_y = Min(y2, y4);

                                int width = right_down_x - left_up_x;
                                int height = right_down_y - left_up_y;

                                int iCrossarea = width * height;

                                if (iIarea > iJarea)
                                {
                                    if ((double)(iCrossarea) / (double)(iIarea) * 100 > 60.0 || (double)(iCrossarea) / (double)(iJarea) * 100 > 60.0)
                                    {                                                         
                                        RectList.RemoveAt(j);                                       
                                        i = 0;                                                
                                                                                              
                                        if (j == RectList.Count)                              
                                            --j;                                              
                                    }                                                         
                                }                                                             
                                else if(iIarea < iJarea)                                      
                                {                                                             
                                    if ((double)(iCrossarea) / (double)(iJarea) * 100 > 70.0 || (double)(iCrossarea) / (double)(iIarea) * 100 > 70.0)
                                    {
                                        RectList.RemoveAt(i);
                                        j = 0;

                                        if (i == RectList.Count)
                                            --i;
                                    }
                                }                          

                            }
                        }
                    }
                    for (int i = 0; i < RectList.Count(); ++i)
                    {
                        Scalar sVariableColor = Scalar.Red;
                        ChangeColor(RectList[i].iIdx, ref sVariableColor);
                        Cv2.Rectangle(listImage[listnum][iRefIdx], RectList[i].Rrect, sVariableColor, 2);
                    }


                    Cv2.Resize(listImage[listnum][iRefIdx], listImage[listnum][iRefIdx], new Size(800, 600));
                    Cv2.ImShow("Image", listImage[listnum][iRefIdx]);
                    Cv2.WaitKey(0);
                    

                    //Fourrulecalculations(listImage[listnum][0], listImage[listnum][iRefIdx]); 연습코드

                    //Mat cutROIIamge = RIOImage.SubMat(boundingRect);
                    //Cv2.Resize(cutROIIamge, dst, new Size(800, 600));
                    //Cv2.ImShow("image", dst);
                    //Cv2.WaitKey(0);
                    //Cv2.DestroyAllWindows();         
                }
            }
        }
    }
}
