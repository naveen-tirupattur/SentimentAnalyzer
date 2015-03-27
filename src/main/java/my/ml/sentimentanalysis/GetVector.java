package my.ml.sentimentanalysis;

import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.apache.commons.lang.StringUtils;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.inputsanitation.InputHomogenization;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;

import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.DocumentPreprocessor.DocType;

public class GetVector {

	@SuppressWarnings({ "rawtypes", "unchecked" })
	public static INDArray createVector(String key, String documentText, Word2Vec wordVector, int numOfFeatures) {

		
		String temp = new InputHomogenization(documentText).transform();
		
		Collection<String> sentences = new ArrayList<String>();
		Reader reader = new StringReader(temp);
		DocumentPreprocessor sentencesList = new DocumentPreprocessor(reader, DocType.Plain);
		for(List sentence:sentencesList) {
			temp = TextUtils.removeTags(sentence);
			sentences.add(temp);
		}
		
		String cleanedText = StringUtils.join(sentences.toArray());
		sentences = null;
		
		//Tokenize the sentence and remove stop words
		TokenizerFactory tokenizerFactory = TextUtils.getTokenizerFactory(true);
		if(cleanedText == null || cleanedText.length()==0) return null;
		Tokenizer tokenizer =  tokenizerFactory.create(cleanedText);
		List<String> tokens = tokenizer.getTokens();
		
		int wordCount = 0;
		INDArray wordWeights = wordVector.getWordVectorMatrix(tokens.get(0));
		
		//Iterate over each token and get the vector of each token from word2vec
		for(int i =1;i<tokens.size();i++) {
			String token = tokens.get(i);
			if(wordVector.hasWord(token)) {
				wordWeights.addi(wordVector.getWordVectorMatrix(token));
				wordCount++;
			}
			token = null;
		}
		tokens = null;
		INDArray normalizedWordWeights = wordWeights.divi(Integer.valueOf(wordCount));
		if(Arrays.toString(normalizedWordWeights.ravel().data().asFloat()).contains("Infinity")) {
			System.out.println(cleanedText);
		}
		return normalizedWordWeights;
		
	}

}
