using System;
using System.IO;
using System.Globalization;

namespace Isaque_BP
{
    class Program
    {
        static void Main(string[] args)
        {
            //Rede Neural
            const int numInputs = 4;
            const int neuronsHidden = 5;
            const int neuronsOut = 3;
            const double learningRate = 0.09;

            double[,] wInpHid = new double[neuronsHidden, numInputs];
            double[,] wHidOut = new double[neuronsOut, neuronsHidden];

            Random random = new Random((int)DateTime.UtcNow.Ticks);

            //inicializando as matrizes de pesos
            Console.WriteLine("Pesos iniciais entrada-oculta:\n");
            for (int i = 0; i < wInpHid.GetLength(0); ++i)
            {
                for (int j = 0; j < wInpHid.GetLength(1); ++j)
                {
                    wInpHid[i, j] = random.NextDouble();
                    Console.Write(wInpHid[i, j] + " ");
                }
                Console.WriteLine();
            }
            Console.WriteLine("\nPesos iniciais oculta-saida:\n");
            for (int i = 0; i < wHidOut.GetLength(0); ++i)
            {
                for (int j = 0; j < wHidOut.GetLength(1); ++j)
                {
                    wHidOut[i, j] = random.NextDouble();
                    Console.Write(wHidOut[i, j] + " ");
                }
                Console.WriteLine();
            }

            //Arquivos
            string fileTest, fileTrain;
            fileTrain = Directory.GetCurrentDirectory() + "\\..\\..\\..\\..\\iris.txt";
            fileTest = Directory.GetCurrentDirectory() + "\\..\\..\\..\\..\\iris2.txt";

            string[] trainFileInputs = File.ReadAllLines(fileTrain);
            double[,] trainInputsFile = DoubleInputs(trainFileInputs);
            double[] expectedFlowers = ExpectedResult(trainFileInputs);

            //Condições de parada
            int numGens = 50000;
            double cost = 0.00001;
            double percentage = 0.9;

            TrainANN();

            void TrainANN()
            {
                Console.WriteLine("\nTreinando");
                int currentGen = 0;
                int numCosts = 0;

                while (true)
                {
                    for (int i = 0; i < trainFileInputs.Length; ++i)
                    {
                        //Console.WriteLine(i);
                        double[] resultHidden = new double[neuronsHidden];
                        double[] resultOutput = new double[neuronsOut];
                        double[] outputErrors = new double[neuronsOut];
                        double[] derivatesOutput = new double[neuronsOut];
                        double[] derivatesHidden = new double[neuronsHidden];
                        double sumOutputErrors = 0.0;
                        
                        //Algoritmo - FeedForward
                        for (int j = 0; j < neuronsHidden; ++j)
                        {
                            resultHidden[j] = OutputPerceptron(MatrixLineToArray(trainInputsFile, i), MatrixLineToArray(wInpHid, j));
                        }
                        for (int j = 0; j < neuronsOut; ++j)
                        {
                            resultOutput[j] = OutputPerceptron(resultHidden, MatrixLineToArray(wHidOut, j));
                            outputErrors[j] = expectedFlowers[i] - resultOutput[j];
                            derivatesOutput[j] = resultOutput[j] * (1.0 - resultOutput[j]);
                            sumOutputErrors += outputErrors[j] * derivatesOutput[j];
                        }

                        if (SquaredError(outputErrors) <= cost)
                        {
                            Console.WriteLine(SquaredError(outputErrors));
                            ++numCosts;
                        }

                        //Algoritmo - Backpropagation
                        //Pesos oculta-saida
                        for (int j = 0; j < neuronsOut; ++j)
                        {
                            //Console.WriteLine(neuronsOut);
                            double error = outputErrors[j] * derivatesOutput[j];
                            for (int a = 0; a < wHidOut.GetLength(1); ++a)
                            {
                                wHidOut[j, a] += learningRate * error * resultHidden[a];
                            }
                        }

                        //Pesos entrada-oculta
                        for (int j = 0; j < neuronsHidden; ++j)
                        {
                            derivatesHidden[j] = resultHidden[j] * (1.0 - resultHidden[j]);
                            double error = derivatesHidden[j] * sumOutputErrors;

                            for (int a = 0; a < wInpHid.GetLength(1); ++a)
                            {
                                wInpHid[j, a] += learningRate * error * trainInputsFile[i, a];
                            }
                        }
                        ++currentGen;
                    }
                    if (numCosts >= trainFileInputs.Length * percentage)
                    {
                        break;
                    }
                    else if (currentGen > numGens)
                    {
                        //inicializando as matrizes de pesos
                        Console.WriteLine("Pesos iniciais entrada-oculta:\n");
                        for (int i = 0; i < wInpHid.GetLength(0); ++i)
                        {
                            for (int j = 0; j < wInpHid.GetLength(1); ++j)
                            {
                                wInpHid[i, j] = random.NextDouble();
                                Console.Write(wInpHid[i, j] + " ");
                            }
                            Console.WriteLine();
                        }
                        Console.WriteLine("\nPesos iniciais oculta-saida:\n");
                        for (int i = 0; i < wHidOut.GetLength(0); ++i)
                        {
                            for (int j = 0; j < wHidOut.GetLength(1); ++j)
                            {
                                wHidOut[i, j] = random.NextDouble();
                                Console.Write(wHidOut[i, j] + " ");
                            }
                            Console.WriteLine();
                        }

                        TrainANN();
                        break;
                    }
                }
            }

            //Mostrando as matrizes de pesos treinadas
            Console.WriteLine("Pesos treinados entrada-oculta:\n");
            for (int i = 0; i < wInpHid.GetLength(0); ++i)
            {
                for (int j = 0; j < wInpHid.GetLength(1); ++j)
                {
                    Console.Write(wInpHid[i, j] + " ");
                }
                Console.WriteLine();
            }
            Console.WriteLine("\nPesos treinados oculta-saida:\n");
            for (int i = 0; i < wHidOut.GetLength(0); ++i)
            {
                for (int j = 0; j < wHidOut.GetLength(1); ++j)
                {
                    Console.Write(wHidOut[i, j] + " ");
                }
                Console.WriteLine();
            }

            //Lendo arquivo de teste
            string[] testFileInputs = File.ReadAllLines(fileTest);
            double[,] testInputsFile = DoubleInputs(testFileInputs);

            double[] testResultHidden = new double[neuronsHidden];
            double[] testResultOutput = new double[neuronsOut];

            for (int i = 0; i < testFileInputs.Length; ++i)
            {
                for (int j = 0; j < neuronsHidden; ++j)
                {
                    testResultHidden[j] = OutputPerceptron(MatrixLineToArray(testInputsFile, i), MatrixLineToArray(wInpHid, j));
                }
                for (int j = 0; j < neuronsOut; ++j)
                {
                    testResultOutput[j] = OutputPerceptron(testResultHidden, MatrixLineToArray(wHidOut, j));
                }

                Console.WriteLine("Flor: " + ConvertNumberToFlower(testResultOutput[0]) + " " + testResultOutput[0]);
            }

            //Calculo do perceptron
            double OutputPerceptron(double[] pInputs, double[] pWeights)
            {
                double output = 0.0;
                double sum = 0.0f;

                for (int i = 0; i < pInputs.Length; ++i)
                {
                    sum += pInputs[i] * pWeights[i];
                }

                output = 1.0 / (1.0 + Math.Exp(-sum)); //sigmoid

                return output;
            }

            double[,] DoubleInputs(string[] fileRead)
            {
                NumberFormatInfo separator = (NumberFormatInfo)CultureInfo.CurrentCulture.NumberFormat.Clone();
                separator.NumberDecimalSeparator = ".";

                int h = fileRead[0].Split(",").Length - 1;
                double[,] matrixReturn = new double[fileRead.Length, h];

                for (int i = 0; i < fileRead.Length; ++i)
                {
                    for (int j = 0; j < h; ++j)
                    {
                        matrixReturn[i, j] = double.Parse(fileRead[i].Split(",")[j], separator);
                    }
                }

                return matrixReturn;
            }

            double[] ExpectedResult(string[] flowerNames)
            {
                double[] arrayReturn = new double[flowerNames.Length];

                int h = flowerNames[0].Split(",").Length - 1;

                for (int i = 0; i < flowerNames.Length; ++i)
                {
                    switch (flowerNames[i].Split(",")[h])
                    {
                        case "Iris-setosa":
                            arrayReturn[i] = 0.0;
                            break;
                        case "Iris-versicolor":
                            arrayReturn[i] = 0.5;
                            break;
                        case "Iris-virginica":
                            arrayReturn[i] = 1.0;
                            break;
                    }
                }

                return arrayReturn;
            }

            double SquaredError(double[] errors)
            {
                double squaredError = 0.0;

                foreach(double error in errors)
                {
                    squaredError += error * error;
                }

                return squaredError;
            }

            string ConvertNumberToFlower(double value)
            {
                if (value < 0.3333334)
                {
                    return "Iris-setosa";
                }
                else if (value > 0.6666667)
                {
                    return "Iris-virginica";
                }
                else
                {
                    return "Iris-versicolor";
                }
            }

            double[] MatrixLineToArray(double[,] matrix, int line)
            {
                double[] lineToReturn = new double[matrix.GetLength(1)];

                for (int i = 0; i < matrix.GetLength(1); ++i)
                {
                    lineToReturn[i] = matrix[line, i];
                }

                return lineToReturn;
            }

            Console.ReadKey();
        }
    }
}