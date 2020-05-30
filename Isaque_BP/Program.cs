using System;

namespace Isaque_BP
{
    class Program
    {

        static void Main(string[] args)
        {
            int numOfInputs;
            double[] userInputs, userWeights;
            double bias, output = 0.0, sumInputs = 0.0;

            Console.WriteLine("Escreva quantas entradas você quer:");
            numOfInputs = int.Parse(Console.ReadLine());
            userInputs = new double[numOfInputs];
            userWeights = new double[numOfInputs];

            Console.WriteLine("Escreva os valores dos inputs separados por espaço:");
            string[] userDesiredInputs = Console.ReadLine().Split();
            Console.WriteLine("Escreva os valores dos pesos separados por espaço:");
            string[] userDesiredWeights = Console.ReadLine().Split();
            Console.WriteLine("Escreva seu valor do bias:");
            bias = double.Parse(Console.ReadLine());

            //Preencher os vetores com os inputs e os pesos
            for (int i = 0; i < numOfInputs; ++i)
            {
                userInputs[i] = double.Parse(userDesiredInputs[i]);
            }

            for (int i = 0; i < numOfInputs; ++i)
            {
                userWeights[i] = double.Parse(userDesiredWeights[i]);
            }

            //Somatória
            for (int i = 0; i < numOfInputs; ++i)
            {
                sumInputs += userInputs[i] * userWeights[i];
            }

            //Bias
            sumInputs -= bias;

            Sigmoid(sumInputs);

            Console.WriteLine("Output: " + output);

            //Função de ativação (Sigmoide)
            void Sigmoid(double exp)
            {
                output = 1 / (1 + Math.Exp(-sumInputs)); //1 / 1 + E ^ -x
            }
        }

        //Etapas do perceptron
        //1. Pegar os inputs e os pesos
        //2. Fazer a somatória do resultado da multiplicação do input pelo seu respectivo peso
        //3. Descontar o bias
        //4. Manda pra função de ativação (sigmoide) =====> Output
    }
}
