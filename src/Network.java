import java.util.*;

/*
 * Aditya Tadimeti
 * February 26, 2020
 * The purpose of this project is to implement the XOR spreadsheet code in Java using a configurable neural network.
 */
public class Network
{
   private int[] hiddenLayerSizes;           // this is an array of the number of nodes in each hidden layer
   private static int[] layerSizes;          // this is an array of the lengths of each layer
   private static int outputNodes;           // this is the number of output nodes
   private static int inputNodes;            // this is the number of input nodes
   private static double[][][] weights;      // this is a 3D array containing all the weight values
   private static double[][] activations;    // this is a 2D array containing all the activations for the network
   private static int activationsSize;       // this is the width of the network
   private static int weightsLength;         // this is the number of layers of weights
   private static int hiddenLayerNodesSize;  // this is the number of hidden layers
   private static final int NUM_OUTPUTS = 1; // this is the number of outputs
   private static double[][] thetaValues;    // this is a 2D array containing all the theta values for the network
   private static double[][][] deltaWeights; // this is a 3D array containing the delta values for each weight in the network
   private static int iterations;            // this is the number of times the network iterates during training

   /*
    * The Network constructor is used to construct neural network objects of type Network to solve the XOR problem.
    * @param inputNodes is the number of nodes the network contains in the first layer, or the input layer. This is user-specified.
    * @param hiddenLayerSizes is an array containing the number of nodes in each hidden layer.
    * @param outputNodes is the number of output nodes the network contains in the final layer. This is also user-specified.
    */
   public Network(int inputNodes, int[] hiddenLayerSizes, int outputNodes)
   {
      this.inputNodes = inputNodes;
      this.outputNodes = outputNodes;

      this.hiddenLayerSizes = hiddenLayerSizes;
      hiddenLayerNodesSize = hiddenLayerSizes.length;

      weightsLength = hiddenLayerNodesSize + 1;
      weights = new double[weightsLength][][];
      deltaWeights = new double[weightsLength][][];
      initializeWeights();

      activationsSize = hiddenLayerNodesSize + 2;
      activations = new double[activationsSize][];
      thetaValues = new double[activationsSize-1][];

      initializeActivationsAndThetas(inputNodes, outputNodes, hiddenLayerNodesSize);

      layerSizes = new int[activationsSize];
      layerSizes[0] = inputNodes;
      layerSizes[activationsSize - 1] = outputNodes;
      for (int hiddenLayerIndex = 0; hiddenLayerIndex < hiddenLayerNodesSize; hiddenLayerIndex++)
      {
         int layerSizesIndex = hiddenLayerIndex + 1;
         layerSizes[layerSizesIndex] = hiddenLayerSizes[hiddenLayerIndex];
      }

      iterations = 0;

   }

   /*
    * This method generates a random number from a given lower bound and upper bound.
    * @return a random number within a lower and upper bound.
    */
   public static double generateRandomWeightValue(double lowerBound, double upperBound)
   {
     return Math.random()*(upperBound-lowerBound)+lowerBound;
   }

   /*
    * The randomizeWeights method sets all the weight values in the network to random values from a user-specified range.
    * @param lowerBound is the lower bound of the random doubles that the user specifies.
    * @param upperBound is the upper bound of the random doubles that the user specifies
    */
   public static void randomizeWeights(double lowerBound, double upperBound)
   {
      for (int layer = 0; layer < weightsLength; layer++)
      {
         for (int currentLayerNode = 0; currentLayerNode < layerSizes[layer]; currentLayerNode++)
         {
            for(int nextLayerNode = 0; nextLayerNode < layerSizes[layer+1]; nextLayerNode++)
            {
               weights[layer][currentLayerNode][nextLayerNode] = generateRandomWeightValue(lowerBound, upperBound);
            }
         }
      } // for (int layer = 0; layer < weightsLength; layer++)
   }

   /*
    * This method initializes the activations and theta array such that they become 2D jagged arrays with all the proper dimensions.
    * @param inputNodes is the number of input nodes
    * @param outputNodes is the number of output nodes
    * @param hiddenLayerNodesSize is the number of hidden layers
    */
   public void initializeActivationsAndThetas(int inputNodes, int outputNodes, int hiddenLayerNodesSize)
   {
      activations[0] = new double[inputNodes];
      activations[hiddenLayerNodesSize+1] = new double[outputNodes];
      thetaValues[hiddenLayerNodesSize] = new double[outputNodes];

      for (int layer = 1; layer < hiddenLayerNodesSize+1; layer++)
      {
         activations[layer] = new double[hiddenLayerSizes[layer-1]];
         thetaValues[layer-1] = new double[hiddenLayerSizes[layer-1]];
      }

   }



   /*
    * This method initializes the weights array such that it becomes a 3D jagged array with the proper dimensions.
    */
   public void initializeWeights()
   {
      weights[0] = new double[inputNodes][hiddenLayerSizes[0]];
      deltaWeights[0] = new double[inputNodes][hiddenLayerSizes[0]];

      weights[hiddenLayerNodesSize] = new double[hiddenLayerSizes[hiddenLayerNodesSize-1]][outputNodes];
      deltaWeights[hiddenLayerNodesSize] = new double[hiddenLayerSizes[hiddenLayerNodesSize-1]][outputNodes];

      for (int layer = 1; layer < hiddenLayerNodesSize; layer++)
      {
         weights[layer] = new double[hiddenLayerSizes[layer-1]][hiddenLayerSizes[layer]];
         deltaWeights[layer] = new double[hiddenLayerSizes[layer-1]][hiddenLayerSizes[layer]];
      }
   }

   /*
    * This method sets the network's activations that are not input activations using dot products and weights.
    */
   public static void propagate()
   {
      for (int layer = 1; layer < activationsSize; layer++)
      {
         for (int currentLayerNode = 0; currentLayerNode < layerSizes[layer]; currentLayerNode++)
         {
            activations[layer][currentLayerNode] = 0.0;
            for (int previousLayerNode = 0; previousLayerNode < layerSizes[layer-1]; previousLayerNode++)
            {
               activations[layer][currentLayerNode] += activations[layer-1][previousLayerNode] * weights[layer-1][previousLayerNode][currentLayerNode];
            }
            thetaValues[layer-1][currentLayerNode] = activations[layer][currentLayerNode];
            activations[layer][currentLayerNode] = thresholdFunction(thetaValues[layer-1][currentLayerNode]);


         }
      } // for (int layer = 1; layer < activationsSize; layer++)
   }

   /*
    * This method writes the threshold function that defines the output of a node given a set of inputs. It is currently f(x) = x.
    * @param input is the input to the node.
    * @return the output of the node, which is a modified version of the input based on the threshold function.
    */
   public static double thresholdFunction(double input)
   {
      return 1.0/(1.0+Math.exp(-input)); //can be customizable later
   }

   /*
    * This method calculates the error based on the network's output and the user's expected output using a least squares method.
    * @param expectedOutput is the user's expected output.
    * @return the value of the error
    */
   public static double calculateError(double expectedOutput, double output)
   {
      double difference = output - expectedOutput;
      return 0.5*difference*difference;
   }


   /**
    *
    * @param input
    * @return
    */
   public static double derivative(double input)
   {
      return thresholdFunction(input)*(1.0-thresholdFunction(input));
   }
   /*
    * This method asks the user for the number of input activations to the network.
    * @return an integer value of the number of inputs the user desires.
    */
   public static int askForNumInputActivations()
   {
      Scanner scanner = new Scanner(System.in);
      System.out.println("How many input activations would you like?");

      return scanner.nextInt();
   }

   /*
    * This method asks the user both the number of hidden layers and the number of nodes in each hidden layer.
    * @return an array of integers containing the number of nodes in each hidden layer as its elements.
    */
   public static int[] askForHiddenLayers()
   {
      Scanner scanner = new Scanner(System.in);
      System.out.println("How many hidden layers would you like?");
      int numHiddenLayers = scanner.nextInt();

      int[] hiddenLayerSizes = new int[numHiddenLayers];
      for (int hiddenLayer = 0; hiddenLayer < numHiddenLayers; hiddenLayer++)
      {
         System.out.println("How many nodes would you like in hidden layer " + hiddenLayer +
                            "? Note: Hidden Layer 0 refers " +
                            "to the first hidden layer, and the rest of the numbers follow the same rule.");
         hiddenLayerSizes[hiddenLayer] = scanner.nextInt();
      }
      return hiddenLayerSizes;
   }

   /*
    * This method asks the user for the values they would like for each of the weights in the network.
    */
   public static void getUserWeights()
   {
      Scanner scanner = new Scanner(System.in);
      for (int layer = 0; layer < weightsLength; layer++)
      {
         for (int previousLayerNode = 0; previousLayerNode < layerSizes[layer]; previousLayerNode++)
         {
            for (int nextLayerNode = 0; nextLayerNode < layerSizes[layer+1]; nextLayerNode++)
            {
               System.out.println("What value would you like for weight[" + layer + "][" + previousLayerNode + "][" + nextLayerNode + "]");
               double weightValue = scanner.nextDouble();
               weights[layer][previousLayerNode][nextLayerNode] = weightValue;
            }
         }
      } // for (int layer = 0; layer < weightsLength; layer++)
   }

   /*
    * This method prints out all the values for the activations in the network.
    */
   public static void printActivations()
   {
      for (int layer = 0; layer < hiddenLayerNodesSize+1; layer++)
      {
         for (int node = 0; node < layerSizes[layer]; node++)
         {
            System.out.println("Activation[" + layer + "]" + "[" + node + "]" + ": " + activations[layer][node]);
         }
      }
   }

   /*
    * This method prints out all the values for the weights in the network.
    */
   public static void printWeights()
   {
      for (int layer = 0; layer < weightsLength; layer++)
      {
         for (int previousLayerNodes = 0; previousLayerNodes < layerSizes[layer]; previousLayerNodes++)
         {
            for (int nextLayerNodes = 0; nextLayerNodes < layerSizes[layer+1]; nextLayerNodes++)
            {
               System.out.println("weight[" + layer + "][" +
                                  previousLayerNodes + "][" +
                                  nextLayerNodes + "]: " +
                                  weights[layer][previousLayerNodes][nextLayerNodes]);
            }
         }
      } // for (int layer = 0; layer < weightsLength; layer++)
   }

   public static void calculateHiddentoOutputDeltaWeights(double thetaZero, double omegaZero, double littlePsiZero, double learningFactor)
   {
      for(int hiddenLayer = 1; hiddenLayer <= hiddenLayerNodesSize; hiddenLayer++)
      {
         for(int hiddenLayerNodeIndex = 0; hiddenLayerNodeIndex < layerSizes[hiddenLayer]; hiddenLayerNodeIndex++)
         {
            double gradient = -activations[hiddenLayer][hiddenLayerNodeIndex]*littlePsiZero;
            double deltaWeight = -learningFactor*gradient;
            deltaWeights[hiddenLayer][hiddenLayerNodeIndex][0] = deltaWeight;
         }
      }
   }

   public static void calculateInputtoHiddenDeltaWeights(double omegaZero, double littlePsiZero, double learningFactor)
   {
      for(int inputLayerNodeIndex = 0; inputLayerNodeIndex < inputNodes; inputLayerNodeIndex++)
      {
         for(int hiddenLayerNodeIndex = 0; hiddenLayerNodeIndex < layerSizes[1]; hiddenLayerNodeIndex++)
         {
            double thetaJ = thetaValues[0][hiddenLayerNodeIndex]; //this is the first activation in the hidden layer, without the threshold function
            double omega = littlePsiZero*weights[weightsLength-1][hiddenLayerNodeIndex][NUM_OUTPUTS-1];
            double psi = omega*derivative(thetaJ);
            double gradient = -activations[0][inputLayerNodeIndex]*psi;
            double deltaWeight = -learningFactor*gradient;
            deltaWeights[0][inputLayerNodeIndex][hiddenLayerNodeIndex] = deltaWeight;
         }
      }
   }

   public static void applyWeights()
   {
      for (int layer = 0; layer < weightsLength; layer++)
      {
         for (int currentLayerNode = 0; currentLayerNode < layerSizes[layer]; currentLayerNode++)
         {
            for(int nextLayerNode = 0; nextLayerNode < layerSizes[layer+1]; nextLayerNode++)
            {
               weights[layer][currentLayerNode][nextLayerNode]+= deltaWeights[layer][currentLayerNode][nextLayerNode];
            }
         }
      } // for (int layer = 0; layer < weightsLength; layer++)
   }
   public static void train(double expectedOutput, double learningFactor)
   {
      double thetaZero = thetaValues[activationsSize-2][NUM_OUTPUTS-1];
      double omegaZero = expectedOutput-thresholdFunction(thetaZero);
      double littlePsiZero = derivative(thetaZero)*omegaZero;
      calculateHiddentoOutputDeltaWeights(thetaZero, omegaZero, littlePsiZero, learningFactor);
      calculateInputtoHiddenDeltaWeights(omegaZero, littlePsiZero, learningFactor);
      applyWeights();
   }

   public static void askForNetworkConfiguration()
   {
      int numInputs = askForNumInputActivations();
      int[] hiddenLayerSizes = askForHiddenLayers();

      Scanner scanner = new Scanner(System.in);
      double[] inputActivations = new double[numInputs];

      Network network = new Network(numInputs, hiddenLayerSizes, NUM_OUTPUTS);
      activations[0] = inputActivations;

      System.out.println("Type 1 to set the weights manually, or type 2 to randomize them.");
      int answer = scanner.nextInt();
      if (answer == 1)
      {
         getUserWeights();
      }
      else if (answer == 2)
      {
         System.out.println("What is the lower bound of the random numbers you would like?");
         double lowerBound = scanner.nextDouble();
         System.out.println("What is the upper bound of the random numbers you would like?");
         double upperBound = scanner.nextDouble();
         randomizeWeights(lowerBound,upperBound);
      }
      printWeights();

   }


   public static double[] askForTrainingSetConfiguration(int trainingSetNumber)
   {
      Scanner scanner = new Scanner(System.in);

      System.out.println("");
      System.out.println("For Training Set " + trainingSetNumber + ":");
      double[] trainingSetConfiguration = new double[inputNodes+1];

      for (int inputActivationIndex = 0; inputActivationIndex < inputNodes; inputActivationIndex++) {
         System.out.println("Please enter the value for activation[0][" + inputActivationIndex + "]");
         trainingSetConfiguration[inputActivationIndex] = scanner.nextDouble();
      }
      System.out.println("What is the expected output?");
      trainingSetConfiguration[inputNodes] = scanner.nextDouble();

      return trainingSetConfiguration;
   }

   public static void executeNetwork(double learningFactor, double errorThreshold, double expectedOutput, int maxIterations)
   {
      propagate();
      System.out.println("");

      train(expectedOutput, learningFactor);
      propagate();
      System.out.println(iterations);
      iterations++;

   }

   public static boolean determineNetworkState(double[] trainingSetErrors, double errorThreshold, int totalTrainingSets, int maxIterations)
   {
      boolean networkState = true;
      for(int trainingSet = 0; trainingSet < totalTrainingSets; trainingSet++)
      {
         if(trainingSetErrors[trainingSet]>errorThreshold)
         {
            networkState = false;
         }
      }
      if(iterations > maxIterations)
      {
         networkState = true;
      }
      return networkState;
   }

   /*
    * This method is run to evaluate the network.
    * Users must input valid numbers when prompted.
    * @param args is an array of string inputs for the command line.
    */
   public static void main(String[] args)
   {
      Scanner scanner = new Scanner(System.in);

      System.out.println("How many training sets would you like?");
      int totalTrainingSets = scanner.nextInt();

      System.out.println("What would you like the learning factor to be?");
      double learningFactor = scanner.nextDouble();

      System.out.println("What would you like the error threshold to be?");
      double errorThreshold = scanner.nextDouble();

      System.out.println("What would you like the maximum number of iterations to be?");
      int maxIterations = scanner.nextInt();

      askForNetworkConfiguration();
      double[][] trainingSetsInformation = new double[totalTrainingSets][inputNodes];
      for(int trainingSet = 0; trainingSet < totalTrainingSets; trainingSet++)
      {
         trainingSetsInformation[trainingSet] = askForTrainingSetConfiguration(trainingSet);
      }

      double errorsForTrainingSets[] = new double[totalTrainingSets];
      for(int error = 0; error < totalTrainingSets; error++)
      {
         errorsForTrainingSets[error] = Double.MAX_VALUE;
      }

     while(!determineNetworkState(errorsForTrainingSets, errorThreshold, totalTrainingSets, maxIterations))
      {
         for (int trainingSet = 0; trainingSet < totalTrainingSets; trainingSet++)
         {
            for (int inputNode = 0; inputNode < inputNodes; inputNode++)
            {
               activations[0][inputNode] = trainingSetsInformation[trainingSet][inputNode];
            }
            executeNetwork(learningFactor, errorThreshold, trainingSetsInformation[trainingSet][inputNodes], maxIterations);
            errorsForTrainingSets[trainingSet] = calculateError(activations[hiddenLayerNodesSize+1][0],trainingSetsInformation[trainingSet][inputNodes]);
         }
      }

      System.out.println("");
      System.out.println("Below are the final weights used in the network following training:");
      printWeights();
      System.out.println("");

      for(int trainingSet = 0; trainingSet < totalTrainingSets; trainingSet++)
      {
         System.out.println("The error for Training Set " + trainingSet + " was " + errorsForTrainingSets[trainingSet]);
      }

      System.out.println("The total number of iterations was " + iterations);

      for(int trainingSet = 0; trainingSet < totalTrainingSets; trainingSet++)
      {
         System.out.println("The expected output for Training Set " + trainingSet + " was " + trainingSetsInformation[trainingSet][inputNodes]);
         for(int inputNode = 0; inputNode < inputNodes; inputNode++)
         {
            activations[0][inputNode] = trainingSetsInformation[trainingSet][inputNode];
         }
         propagate();
         System.out.println("The actual output for Training Set " + trainingSet + "was " + activations[activationsSize-1][0]);
      }
   }

}
























