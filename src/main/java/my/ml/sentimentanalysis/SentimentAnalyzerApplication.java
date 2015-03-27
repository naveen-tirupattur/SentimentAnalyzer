package my.ml.sentimentanalysis;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.core.io.ClassPathResource;


public class SentimentAnalyzerApplication {

	@SuppressWarnings({ "rawtypes","unchecked" })
	public static void main(String args[]) {

		long startTime = System.currentTimeMillis();

		int numOfFeatures = 300;

		//generate the wordvector from the data
		//GenerateWordVector.createWordVector("/labeledTrainData.tsv",numOfFeatures);

		//read the wordvector
		
		try {
			ClassPathResource resource = new ClassPathResource("/wordVectorWithStopWords");
			Word2Vec wordVector = SerializationUtils.readObject(resource.getInputStream());
			//			Collection<String> relatedWords = wordVector.wordsNearest("cool", 10);
			//		System.out.println(relatedWords);
			
			//Write the vectors to file
			//CSVWriter writer = new CSVWriter(new FileWriter(new File("trainingData.csv"),true), ',');
			List<Pair> data = new ArrayList<Pair>();
			//Read the dataset from file
			Map<String, Pair<String,String>> dataMap = TextUtils.getDataMap("/labeledTrainData.tsv");
			System.out.println("Dataset size: "+dataMap.size());
			int count = 0;
			String label = "", documentText = "";
			for(Entry e: dataMap.entrySet() ) {
				String key = (String) e.getKey();
				Pair p = (Pair)e.getValue();
				label = p.getFirst().toString();
				documentText = p.getSecond().toString();
				if(count%1000 == 0) System.out.println("Processed "+count+" records");
				INDArray vector = GetVector.createVector(key, documentText, wordVector, numOfFeatures);
				data.add(new Pair(label,vector));
				label = null;
				documentText = null;
				vector = null;
				count++;
			}	
			TextUtils.writeFile("trainingData.csv", data);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		long totalTimeInMilliSecs = System.currentTimeMillis() - startTime;
		System.out.println("Total time to finish in seconds: "+totalTimeInMilliSecs/(1000));

	}
}
