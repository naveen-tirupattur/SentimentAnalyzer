package my.ml.sentimentanalysis;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.springframework.core.io.ClassPathResource;

import au.com.bytecode.opencsv.CSV;
import au.com.bytecode.opencsv.CSVReadProc;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.uima.resource.ResourceInitializationException;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.inputsanitation.InputHomogenization;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.UimaTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.StringCleaning;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;

import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.DocumentPreprocessor.DocType;
import edu.stanford.nlp.process.StripTagsProcessor;

public class GenerateWordVector {
	
	private static Log log = LogFactory.getLog(GenerateWordVector.class);
	private static Word2Vec wordVector;

	//create the word vector object from the data
	@SuppressWarnings({ "rawtypes", "unchecked"})
	public static void createWordVector(String fileName, int numOfFeatures) {
		//Collection of reviews as sentences
		Collection<String> sentences = new ArrayList<String>();
				
		Map<String, Pair<String,String>> dataMap = TextUtils.getDataMap(fileName);
		System.out.println("Dataset size: "+dataMap.size());
		
		//Read the content of each review and convert it into sentences
		for(Pair p:dataMap.values()) {
			Reader reader = new StringReader(p.getSecond().toString());
			DocumentPreprocessor sentencesList = new DocumentPreprocessor(reader, DocType.Plain);
			for(List sentence:sentencesList) {
				sentences.add(TextUtils.removeTags(sentence));
			}
		}
		
		System.out.println("Number of sentences: "+sentences.size());
		
		//Create the sentence iterator object
		SentenceIterator sentenceIterator = new CollectionSentenceIterator(sentences);
		//Use the pre-processor to remove special characters
		sentenceIterator.setPreProcessor(new SentencePreProcessor() {
			@Override
			public String preProcess(String sentence) {
				String ret = new InputHomogenization(sentence).transform();
				if(ret.isEmpty()) return "a";
				return ret;
			}
		});

		try {
			
			TokenizerFactory t = TextUtils.getTokenizerFactory(true);
			wordVector = new Word2Vec.Builder().sampling(1e-5).workers(10)
					.minWordFrequency(40).batchSize(1000).useAdaGrad(false).layerSize(numOfFeatures)
					.iterations(3).learningRate(0.025).minLearningRate(1e-2).negativeSample(10).windowSize(10)
					.iterate(sentenceIterator).tokenizerFactory(t).build();
			
			wordVector.fit();

			//Write wordvector to file
			SerializationUtils.saveObject(wordVector, new File("myWordVector"));

		} catch (IOException e) {
			log.error(e.getMessage());
		}
	}

}
