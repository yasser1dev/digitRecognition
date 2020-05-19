import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class MainModel {
    public static void main(String[] args) throws IOException, InterruptedException {
        double learningRate=0.001;
        long height=28;
        long width=28;
        long deepth=1;
        int outputSize=10;
        MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Sgd(learningRate))
                .list()
                .setInputType(InputType.convolutionalFlat(height,width,deepth))
                .layer(0,new ConvolutionLayer.Builder()
                                .nIn(deepth)
                                .nOut(20)//filter number to apply
                                .activation(Activation.RELU)
                                .kernelSize(5,5)
                                .stride(1,1)
                                .build()
                        )
                .layer(1,new SubsamplingLayer.Builder()
                                .kernelSize(2,2)
                                .stride(2,2)
                                .poolingType(SubsamplingLayer.PoolingType.MAX)
                                .build())
                .layer(2,new ConvolutionLayer.Builder()
                                .nOut(50)//filter number to apply
                                .activation(Activation.RELU)
                                .kernelSize(5,5)
                                .stride(1,1)
                                .build())
                .layer(3,new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2,2)
                                .stride(2,2)
                                .build())
                .layer(4,new DenseLayer.Builder()
                                .nOut(500)
                                .activation(Activation.RELU)
                                .build())
                .layer(5,new OutputLayer.Builder()
                                .nOut(outputSize)
                                .activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .build())
                .build();

        MultiLayerNetwork model=new MultiLayerNetwork(configuration);
        model.init();
        //System.out.println(configuration.toJson());
        System.out.println("--------- Model training --------");
        int batchSize=54;
        int indexLabel=1;

        File filetrain=new File("C:\\Users\\Lenovo\\Desktop\\BDCC-S4\\SMA\\CNN-digit\\src\\main\\resources\\mnist_png\\mnist_png\\training");
        FileSplit fileSplitTrain=new FileSplit(filetrain, NativeImageLoader.ALLOWED_FORMATS,new Random(123));

        RecordReader recordReaderTrain=new ImageRecordReader(height,width,deepth,new ParentPathLabelGenerator());
        recordReaderTrain.initialize(fileSplitTrain);

        DataSetIterator dataSetTrain=new RecordReaderDataSetIterator(recordReaderTrain,batchSize,indexLabel,outputSize);

        DataNormalization scaler=new ImagePreProcessingScaler(0,1);
        dataSetTrain.setPreProcessor(scaler);
        /*
        while(dataSetTrain.hasNext()){
            DataSet dataSet=dataSetTrain.next();
            INDArray features=dataSet.getFeatures();
            INDArray label=dataSet.getLabels();
            System.out.println(features.shapeInfoToString());
            System.out.println(label);
            System.out.println("-----------");
        }*/
        UIServer uiServer=UIServer.getInstance();
        StatsStorage statsStorage=new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));
        int nEpoch=3;
        for(int i=0;i<nEpoch;i++){
            model.fit(dataSetTrain);
        }

        System.out.println("----- Model Evaluation ----");
        File filetest=new File("C:\\Users\\Lenovo\\Desktop\\BDCC-S4\\SMA\\CNN-digit\\src\\main\\resources\\mnist_png\\mnist_png\\testing");
        FileSplit fileSplitTest=new FileSplit(filetest, NativeImageLoader.ALLOWED_FORMATS);

        RecordReader recordReaderTest=new ImageRecordReader(height,width,deepth,new ParentPathLabelGenerator());
        recordReaderTest.initialize(fileSplitTest);

        DataSetIterator dataSetTest=new RecordReaderDataSetIterator(recordReaderTest,batchSize,indexLabel,outputSize);
        dataSetTest.setPreProcessor(scaler);
        Evaluation evaluation=new Evaluation();
        while(dataSetTest.hasNext()){
            DataSet dataSet=dataSetTest.next();
            INDArray features=dataSet.getFeatures();
            INDArray targetLabels=dataSet.getLabels();
            INDArray predicted=model.output(features);
            evaluation.eval(predicted,targetLabels);
        }

        System.out.println(evaluation.stats());
        System.out.println("---- Model Serialization ");
        //ModelSerializer.writeModel(model,"digitRecognition.zip",true);
    }
}
