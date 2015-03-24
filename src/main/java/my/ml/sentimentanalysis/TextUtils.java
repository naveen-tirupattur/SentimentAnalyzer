package my.ml.sentimentanalysis;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.uima.resource.ResourceInitializationException;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.StringCleaning;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.springframework.core.io.ClassPathResource;

import au.com.bytecode.opencsv.CSV;
import au.com.bytecode.opencsv.CSVReadProc;
import au.com.bytecode.opencsv.CSVWriteProc;
import au.com.bytecode.opencsv.CSVWriter;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.process.StripTagsProcessor;

public class TextUtils {

	private static Log log = LogFactory.getLog(TextUtils.class);

	// define the map to store the labels and reviews
	private static final Map<String, Pair<String,String>> dataMap = new HashMap<String,Pair<String, String>>();

	public static Map<String, Pair<String,String>> getDataMap(String fileName) {
		if(dataMap.isEmpty()) readFile(fileName);
		return dataMap;
	}

	public static void writeFile(String fileName, final Pair<String,double[]> data) {
		//define the file schema
		CSV csv = CSV.separator(',')
				.ignoreLeadingWhiteSpace().create();

		// write CSV file
		csv.write(fileName, new CSVWriteProc() {
			public void process(CSVWriter out) {

				if(data.getFirst() == null || data.getSecond() == null) {
					System.out.println(data.getFirst());
				}
				double[] weights = (double[])data.getSecond();
				String temp = Arrays.toString(weights);
				weights = null;
				out.writeNext(data.getFirst().toString()+","+temp);
				temp = null;

			}
		});
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public static void readFile(String fileName) {
		ClassPathResource resource = new ClassPathResource(fileName);
		//define the file schema
		CSV csv = CSV.separator('\t')
				.ignoreLeadingWhiteSpace().skipLines(1).create();
		try {
			//read the file
			csv.readAndClose(resource.getInputStream(), new CSVReadProc() {
				@Override
				public void procRow(int rowIndex, String... values) {
					//Read the reviews and labels into a map
					dataMap.put(values[0], new Pair(values[1],values[2].toLowerCase())); 
				}
			});
		} catch (IOException e) {
			log.error(e.getMessage());

		}
	}

	public static TokenizerFactory getTokenizerFactory(final boolean removeStopWords) {
		TokenizerFactory t;
		try {
			t = new UimaTokenizerFactory();
			final EndingPreProcessor preProcessor = new EndingPreProcessor();
			final List<String> stopWords = StopWords.getStopWords();
			t.setTokenPreProcessor(new TokenPreProcess() {
				@Override
				public String preProcess(String token) {
					String base = preProcessor.preProcess(token);
					if(removeStopWords) {
						if(stopWords.contains(base)) return "";
					}
					//Remove non-words
					base = base.replaceAll("[^a-z]","");
					return StringCleaning.stripPunct(base);
				}
			});
			return t;
		} catch (ResourceInitializationException e) {
			log.error("Error while creating tokenizer factory: "+e.getMessage());
		}
		return null;
	}

	public static String removeTags(List<Word> text) {
		StripTagsProcessor<Word , Word> stripTags = new StripTagsProcessor<>();
		//Remove HTML tags
		return Sentence.listToString(stripTags.process(text));
	}

}
