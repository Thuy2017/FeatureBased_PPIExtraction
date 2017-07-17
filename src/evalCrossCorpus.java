
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;



public class EvalCrossCorpus{

//	public static String train_corpus = "LLL";
	public static String train_corpus = "HPRD50";
//	public static String train_corpus = "IEPA";
//	public static String train_corpus = "AImed";
//	public static String train_corpus = "BioInfer";

	/*"allExceptLLL" is  the ensemble of four corpora HPRD50, IEPA, AImed, Bioinfer (i.e., exclusion of LLL).
		"allExceptLLL" is used for cross-learning experiments in which we had no time (and no extra page for publication)
		to report the results of these experiment in our paper at ICTAI 2017.
	*/
//	public static String train_corpus = "allExceptLLL";
//	public static String train_corpus = "allExceptHPRD50";
//	public static String train_corpus = "allExceptIEPA";
//	public static String train_corpus = "allExceptAImed";
//	public static String train_corpus = "allExceptBioInfer";


//	public static String test_corpus = "LLL";
//	public static String test_corpus = "HPRD50";
	public static String test_corpus = "IEPA";
//	public static String test_corpus = "AImed";
//	public static String test_corpus = "HPRD50";
	public static int R = 40; //for rand number




	public static void make_rand(int[] rand,int date_num_train){          //ただの乱数作成
		int num = date_num_train*R;
		Random random = new Random(1);
		for(int i = 0;i<num;i++){
			rand[i] = Math.abs(random.nextInt())%date_num_train;
		}
	}


	public static void main(String[] args) throws Exception {

		NumberFormat format = NumberFormat.getInstance();
		format.setMaximumFractionDigits(3);

		int date_num_train = 0;
		int date_num_test = 0;


		if(train_corpus.equals("LLL"))
			date_num_train = 330; //for train_corpus
		else if(train_corpus.equals("HPRD50"))
			date_num_train = 433;
		else if(train_corpus.equals("IEPA"))
			date_num_train = 817;
		else if(train_corpus.equals("AImed"))
			date_num_train = 5834;
		else if(train_corpus.equals("BioInfer"))
			date_num_train = 9666;
		else if(train_corpus.equals("allExceptLLL"))
			date_num_train = 433+817+5834+9666;
		else if(train_corpus.equals("allExceptHPRD50"))
			date_num_train = 330+817+5834+9666; //330+433+817+5834+9666;
		else if(train_corpus.equals("allExceptIEPA"))
			date_num_train = 330+433+5834+9666;
		else if(train_corpus.equals("allExceptAImed"))
			date_num_train = 330+433+817+9666;
		else if(train_corpus.equals("allExceptBioInfer"))
			date_num_train = 330+433+817+5834;

		if(test_corpus.equals("LLL"))
			date_num_test = 330; //for train_corpus
		else if(test_corpus.equals("HPRD50"))
			date_num_test = 433;
		else if(test_corpus.equals("IEPA"))
			date_num_test = 817;
		else if(test_corpus.equals("AImed"))
			date_num_test = 5834;
		else if(test_corpus.equals("BioInfer"))
			date_num_test = 9666;

		DataSource source_train = new DataSource("ARFFFiles20170613/"+ train_corpus +".arff");
		DataSource source_test = new DataSource("ARFFFiles20170613/"+ test_corpus +".arff");

		int random_num_train = date_num_train*R;
		int random_num_test = date_num_test*R;
		Instances instances_train = source_train.getDataSet();  //get data from the file source_train.arf
		Instances instances_test = source_test.getDataSet();	  //get data from the file source_test.arf



		Instances train_A;
		Instances train_B;
		Instances train_C;

		Instances test_A;
		Instances test_B;
		Instances test_C;

		double F = 0;
		double P = 0;
		double R = 0;

		double tp = 0,fp = 0,fn = 0;



		int[] rand_train = new int[random_num_train]; //for train_corpus
		int[] rand_test = new int[random_num_test]; //for test_corpus


 		make_rand(rand_train,date_num_train);  //create random numbers (rand is the array of indexes of instances) //for train_corpus
		make_rand(rand_test,date_num_test);  // //create random numbers (rand is the array of indexes of instances) //for test_corpus



		for(int j = 0;j<random_num_train;j=j+2)   { //shuffle the instances randomly
			instances_train.swap(rand_train[j], rand_train[j+1]); //for train_corpus
		}

		for(int j = 0;j<random_num_test;j=j+2)   { //shuffle the instances randomly
			instances_test.swap(rand_test[j], rand_test[j+1]); //for train_corpus
		}




		Classifier classifierA = new RandomForest();                                  //setting classifier
		Classifier classifierB = new RandomForest();
		Classifier classifierC = new RandomForest();


		classifierA.setOptions(weka.core.Utils.splitOptions("-I 10" + " -K 0 " + "  -S 1"));
		classifierB.setOptions(weka.core.Utils.splitOptions("-I 10" + " -K 0 " + "  -S 1"));
		classifierC.setOptions(weka.core.Utils.splitOptions("-I 10" + " -K 0 " + "  -S 1"));


		instances_train.setClassIndex(instances_train.numAttributes() - 1);     // Make the last attribute be the class //for train_corpus
		instances_test.setClassIndex(instances_test.numAttributes() - 1);      // Make the last attribute be the class //for test_corpus


		/* Check shrink coefficients for each round of the corpus */
		double[] co = new double[6];
		co = CoefficientsofEachRoundCV(10, instances_train, date_num_train, -1);

		double kA = co[0];
		double pA = co[1];
		double kB = co[2];
		double pB = co[3];
		double kC = co[4];
		double pC = co[5];
		System.out.println("Check shrink coefficients for each round of the training corpus: With " + " kA = " +kA  + "; pA = " +pA  + "; kB = " +kB + "; pB = " +pB + "; kC = " +kC  + "; pC = " +pC +";" ) ;
		/*End shrink coefficients for each round of the corpus*/


		train_A = new Instances(instances_train); //shallow copy of instances_train
		train_B = new Instances(instances_train);
		train_C = new Instances(instances_train);

		test_A = new Instances(instances_test);//shallow copy of instances_test
		test_B = new Instances(instances_test);
		test_C = new Instances(instances_test);


		train_A.setClassIndex(train_A.numAttributes() - 1);  // Make the last attribute be the class
		train_B.setClassIndex(train_B.numAttributes() - 1);
		train_C.setClassIndex(train_C.numAttributes() - 1);

		test_A.setClassIndex(test_A.numAttributes() - 1); // Make the last attribute be the class
		test_B.setClassIndex(test_B.numAttributes() - 1);
		test_C.setClassIndex(test_C.numAttributes() - 1);


		//double kA_max = 1, pA_max = 1, kB_max = 1, pB_max = 1, kC_max = 1, pC_max = 1;
	    //kA_max = 1; pA_max = 1; kB_max = 0.9; pB_max =0.9; kC_max =0.9; pC_max =0.9;

		//kA_max = 1.0; pA_max = 0.75; kB_max = 0.75; pB_max = 0.75; kC_max = 1.0; pC_max = 0.75;

		//kA_max = 1.0; pA_max = 1.0; kB_max = 1.0; pB_max = 0.9; kC_max = 0.9; pC_max = 1.0;

		//kA_max = 1.0; pA_max = 1.0; kB_max = 1.0; pB_max = 0.5; kC_max = 0.5; pC_max = 1.0;

		//kA_max = 1.0; pA_max = 1.0; kB_max = 0.8; pB_max = 1.0; kC_max = 1.0; pC_max = 0.8;

		//kA_max = 1.0; pA_max = 1.0; kB_max = 1.0; pB_max = 0.9; kC_max = 1.0; pC_max = 0.9;

		//kA_max = 1.0; pA_max = 1.0; kB_max = 0.8; pB_max = 0.8; kC_max = 0.8; pC_max = 1.0;

		//kA_max = 1.0; pA_max = 1.0; kB_max = 0.0; pB_max = 1.0; kC_max = 0.0; pC_max = 1.0;

		//kA_max = 1.0; pA_max = 1.0; kB_max = 0.0; pB_max = 0.5; kC_max = 0.5; pC_max = 1.0;

/*	    if(train_corpus.equals("AImed")){
	    //	kA = 1.0; pA = 1.0; kB = 0.5; pB = 0.5; kC = 0.5; pC = 1.0;
	    //	kA = 1.0; pA = 0.5; kB = 0.5; pB = 1.0; kC = 0.5; pC = 0.5;
	    	kA = 1.0; pA = 1.0; kB = 1.0; pB = 0.5; kC = 0.5; pC = 0.5; (test=LLL, HPRD50, IEPA, BioInfer)
	    }else if(train_corpus.equals("BioInfer")){
	    //	kA = 1.0; pA = 0.5; kB = 0.5; pB = 1.0; kC = 1.0; pC = 1.0;
	    	kA = 0.5; pA = 1.0; kB = 0.5; pB = 0.5; kC = 0.5; pC = 1.0;
	    }else if(train_corpus.equals("LLL")){
	    	kA = 0.5; pA = 0.5; kB = 1.0; pB = 0.5; kC = 1.0; pC = 1.0;
	    }else if(train_corpus.equals("IEPA")){
	    	kA = 1.0; pA = 0.5; kB = 0.5; pB = 1.0; kC = 0.5; pC = 0.5;
	    }else if(train_corpus.equals("HPRD50")){
	    	kA = 1.0; pA = 0.5; kB = 0.5; pB = 0.5; kC = 1.0; pC = 0.5;
	    }
*/



		instancesCoefficient(kA, pA, kB, pB, kC, pC, train_A);
		instancesCoefficient(kA, pA, kB, pB, kC, pC, train_B);
		instancesCoefficient(kA, pA, kB, pB, kC, pC, train_C);


		instancesCoefficient(kA, pA, kB, pB, kC, pC, test_A);
		instancesCoefficient(kA, pA, kB, pB, kC, pC, test_B);
		instancesCoefficient(kA, pA, kB, pB, kC, pC, test_C);



		set_testABC(instances_test,test_A,test_B,test_C);

		set_trainABC(instances_train,train_A,train_B,train_C);


		/* Feature Selection */
		select_featureB(train_B,test_B);
		select_featureA(train_A,test_A);
		select_featureC(train_C,test_C);

		classifierA.buildClassifier(train_A);
		classifierB.buildClassifier(train_B);
		classifierC.buildClassifier(train_C);


		Evaluation evalA = new Evaluation(train_A);
		Evaluation evalB = new Evaluation(train_B);
		Evaluation evalC = new Evaluation(train_C);


		evalA.evaluateModel(classifierA, test_A);
		evalB.evaluateModel(classifierB, test_B);
		evalC.evaluateModel(classifierC, test_C);


		tp = evalA.numTruePositives(0);
		tp += evalB.numTruePositives(0);
		tp += evalC.numTruePositives(0);

		fp = evalA.numFalsePositives(0);
		fp += evalB.numFalsePositives(0);
		fp += evalC.numFalsePositives(0);

		fn = evalA.numFalseNegatives(0);
		fn += evalB.numFalseNegatives(0);
		fn += evalC.numFalseNegatives(0);


		P = tp / (tp+fp);
		R = tp / (tp+fn);
		F = (2*P*R) / (P + R);

		System.out.println();

		System.out.println("CROSS-CORPUS RESULTS of 3subCC-FSShrink trained on " + train_corpus + " and tested on " +test_corpus +":");
		System.out.println("P = "+ P);
		System.out.println("R = "+ R);
		System.out.println("F = "+ F);

	}

	private static double[] CoefficientsofEachRoundCV(int num_folds, Instances instances, int date_num, int round) throws Exception{

		/* Function Calculation(int kA, int pA, int kB, int pB, int kC, int pC):
		 * kA: coefficient for keyword-related Features in subset A
		 * pA: coefficient for protein-related Features in subset A
		 *
		 * kB: coefficient for keyword-related Features in subset B
		 * pB: coefficient for protein-related Features in subset B
		 *
		 * kC: coefficient for keyword-related Features in subset C
		 * pC: coefficient for protein-related Features in subset C
		 * */


		double F_max, F;
		double kA, pA, kB, pB, kC, pC;
		double kA_max = 1, pA_max = 1, kB_max = 1, pB_max = 1, kC_max = 1, pC_max = 1;
		double DEC = 0.5;
		double MIN = 0.5;

		double arraykA[]= new double[num_folds], arraypA[]=new double[num_folds],arraykB[]=new double[num_folds],arraypB[]=new double[num_folds],arraykC[]=new double[num_folds],arraypC[]=new double[num_folds];

		for(int i=0; i<num_folds; i++){
			arraykA[i]=1; arraypA[i]=1;
			arraykB[i]=1; arraypB[i]=1;
			arraykC[i]=1; arraypC[i]=1;
		}

		F_max = Calculation(num_folds, arraykA, arraypA, arraykB, arraypB, arraykC, arraypC,instances, date_num, false)[2];


//for CV to determnine kA, kB,...
		for(kA=1; kA>=MIN ; kA -=DEC){
			for(pA=1; pA>=MIN ; pA -=DEC){
				for(kB=1; kB>=MIN ; kB -=DEC){
					for(pB=1; pB>=MIN ; pB -=DEC){
						for(kC=1; kC>=MIN ; kC -=DEC){
							for(pC=1; pC>=MIN ; pC -=DEC){
							//	F = Calculation(num_folds,kA, pA, kB, pB, kC, pC,instances, date_num, true)[2];

								for(int i=0; i<num_folds; i++){
									arraykA[i]=kA; arraypA[i]=pA;
									arraykB[i]=kB; arraypB[i]=pB;
									arraykC[i]=kC; arraypC[i]=pC;
								}
								F = Calculation(num_folds, arraykA, arraypA, arraykB, arraypB, arraykC, arraypC,instances, date_num, true)[2];
								if (F > F_max){
									F_max = F;
									kA_max = kA; pA_max = pA; kB_max = kB; pB_max = pB; kC_max = kC; pC_max = pC;
								}
							}
						}
					}
				}
			}
		}


/*		System.out.println("InstanceLevelCV on only ONE CORPUS: On round = "+ round +" of CV: With " + " kA_max = " +kA_max  + "; pA_max = " +pA_max  + "; kB_max = " +kB_max  + "; pB_max = " +pB_max + "; kC_max = " +kC_max  + "; pC_max = " +pC_max +";" ) ;
		System.out.println("F_max = "+F_max);
*/
		double[] r = new double[6];
		r[0]= kA_max;
		r[1]= pA_max;
		r[2]= kB_max;
		r[3]= pB_max;
		r[4]= kC_max;
		r[5]= pC_max;

		return r;

	}



	private static double[] Calculation(int num_folds, double[] kA, double[] pA, double[] kB, double[] pB, double[] kC, double[] pC, Instances instances, int date_num,  boolean FlagAttributeCoefficient) throws Exception {
		/* kA: coefficient for keyword-related Features in subset A
		 * pA: coefficient for protein-related Features in subset A
		 *
		 * kB: coefficient for keyword-related Features in subset B
		 * pB: coefficient for protein-related Features in subset B
		 *
		 * kC: coefficient for keyword-related Features in subset C
		 * pC: coefficient for protein-related Features in subset C
		 * */

		NumberFormat format = NumberFormat.getInstance();
		format.setMaximumFractionDigits(3);


		int random_num = date_num*R;
		//Instances instances = source.getDataSet();                       //データからinstancesを取得

		Instances instances_train;
		Instances instances_test;

		Instances train_A;
		Instances train_B;
		Instances train_C;

		Instances test_A;
		Instances test_B;
		Instances test_C;




		double tp = 0,fp = 0,fn = 0;

		double f=0,p=0,r=0;

		double F = 0;
		double P = 0;
		double R = 0;

		int[] rand = new int[random_num];


	/*	make_rand(rand,date_num);  //create random numbers (rand is the array of indexes of instances)

		for(int j = 0;j<random_num;j=j+2)   {
			instances.swap(rand[j], rand[j+1]);// shuffle the instances randomly
		}
	*/

		Classifier classifierA = new RandomForest();   //setting classifier
		Classifier classifierB = new RandomForest();
		Classifier classifierC = new RandomForest();

		classifierA.setOptions(weka.core.Utils.splitOptions("-I 10" + " -K 0 " + "  -S 1"));
		classifierB.setOptions(weka.core.Utils.splitOptions("-I 10" + " -K 0 " + "  -S 1"));
		classifierC.setOptions(weka.core.Utils.splitOptions("-I 10" + " -K 0 " + "  -S 1"));


		instances.setClassIndex(instances.numAttributes() - 1);        // Make the last attribute be the class



		for(int i = 0;i<num_folds;i++){                                  //9-foldsCV
			instances_train = instances.trainCV(num_folds,i);
			train_A = instances.trainCV(num_folds, i);
			train_B = instances.trainCV(num_folds, i);
			train_C = instances.trainCV(num_folds, i);

			instances_test = instances.testCV(num_folds, i);
			test_A = instances.testCV(num_folds, i);
			test_B = instances.testCV(num_folds, i);
			test_C = instances.testCV(num_folds, i);

			instances_train.setClassIndex(instances_train.numAttributes() - 1);
			train_A.setClassIndex(train_A.numAttributes() - 1);
			train_B.setClassIndex(train_B.numAttributes() - 1);
			train_C.setClassIndex(train_C.numAttributes() - 1);

			instances_test.setClassIndex(instances_test.numAttributes() - 1);
			test_A.setClassIndex(test_A.numAttributes() - 1);
			test_B.setClassIndex(test_B.numAttributes() - 1);
			test_C.setClassIndex(test_C.numAttributes() - 1);



			if(FlagAttributeCoefficient){
					instancesCoefficient(kA[i], pA[i], kB[i], pB[i], kC[i], pC[i], train_A);
					instancesCoefficient(kA[i], pA[i], kB[i], pB[i], kC[i], pC[i], train_B);
					instancesCoefficient(kA[i], pA[i], kB[i], pB[i], kC[i], pC[i], train_C);

					instancesCoefficient(kA[i], pA[i], kB[i], pB[i], kC[i], pC[i], test_A);
					instancesCoefficient(kA[i], pA[i], kB[i], pB[i], kC[i], pC[i], test_B);
					instancesCoefficient(kA[i], pA[i], kB[i], pB[i], kC[i], pC[i], test_C);
				}


			set_testABC(instances_test,test_A,test_B,test_C);

			set_trainABC(instances_train,train_A,train_B,train_C);



			// feature selection//
			select_featureB(train_B,test_B);

			select_featureA(train_A,test_A);
			select_featureC(train_C,test_C);

			classifierA.buildClassifier(train_A);
			classifierB.buildClassifier(train_B);
			classifierC.buildClassifier(train_C);


			Evaluation evalA = new Evaluation(train_A);
			Evaluation evalB = new Evaluation(train_B);
			Evaluation evalC = new Evaluation(train_C);


			evalA.evaluateModel(classifierA, test_A);
			evalB.evaluateModel(classifierB, test_B);
			evalC.evaluateModel(classifierC, test_C);


			tp = evalA.numTruePositives(0);
			tp += evalB.numTruePositives(0);
			tp += evalC.numTruePositives(0);

			fp = evalA.numFalsePositives(0);
			fp += evalB.numFalsePositives(0);
			fp += evalC.numFalsePositives(0);

			fn = evalA.numFalseNegatives(0);
			fn += evalB.numFalseNegatives(0);
			fn += evalC.numFalseNegatives(0);

			p = tp / (tp+fp);
			r = tp / (tp+fn);
			f = (2*p*r) / (p + r);

			P += p;
			R += r;
			F += f;


		}//end for(int i = 0;i<num_folds;i++){


/*		System.out.println();

		System.out.println("InstanceLevelCV on only ONE CORPUS: With " + " kA = " +kA  + "; pA = " +pA  + "; kB = " +kB  + "; pB = " +pB + "; kC = " +kC  + "; pC = " +pC ) ;
		System.out.print("P = "+P/10);
		System.out.print("; R = "+R/10);
		System.out.println("; F = "+F/10);
*/

		double[] results = new double[3];
			results[0]= P/num_folds ;
			results[1]= R/num_folds;
			results[2]= F/num_folds;

			return results;


	}





	private static void select_featureA(Instances train_A, Instances test_A) {
		ArrayList<Integer> delete_ATlist = new ArrayList<Integer>();

		delete_ATlist.add(20); //@attribute pattern_PVP
		delete_ATlist.add(21);//@attribute pattern_PVbyP
		delete_ATlist.add(29);//@attribute pattern_PformcomwithP
		delete_ATlist.add(31);//@attribute pattern_PdependP

		Collections.sort(delete_ATlist);
		Collections.reverse(delete_ATlist);

		for(int i = 0; i < delete_ATlist.size(); i++){
			train_A.deleteAttributeAt(delete_ATlist.get(i));
			test_A.deleteAttributeAt(delete_ATlist.get(i));
		}

	}

	private static void select_featureC(Instances train_C, Instances test_C) {
		ArrayList<Integer> delete_ATlist = new ArrayList<Integer>();

		delete_ATlist.add(20); //@attribute pattern_PVP
		delete_ATlist.add(21);//@attribute pattern_PVbyP
		delete_ATlist.add(29);//@attribute pattern_PformcomwithP
		delete_ATlist.add(31);//@attribute pattern_PdependP

		Collections.sort(delete_ATlist);
		Collections.reverse(delete_ATlist);

		for(int i = 0; i < delete_ATlist.size(); i++){
			train_C.deleteAttributeAt(delete_ATlist.get(i));
			test_C.deleteAttributeAt(delete_ATlist.get(i));
		}


	}


	private static void select_featureB(Instances train_B, Instances test_B) {
		ArrayList<Integer> delete_ATlist = new ArrayList<Integer>();

		delete_ATlist.add(26); //pattern_NbetweenPandP
		delete_ATlist.add(27); //pattern_combetPandP
		delete_ATlist.add(28); // pattern_comofPandP
		delete_ATlist.add(32); //@attribute pattern_between


//		delete_ATlist.add(18);     //overlapF between two proteins P1 and P2
//		delete_ATlist.add(19);     //overlapR between two proteins P1 and P2


		delete_ATlist.add(33);     //pattern_contiguous
		delete_ATlist.add(34);	   //pattern_para


		Collections.sort(delete_ATlist);
		Collections.reverse(delete_ATlist);

		for(int i = 0; i < delete_ATlist.size(); i++){
			train_B.deleteAttributeAt(delete_ATlist.get(i));
			test_B.deleteAttributeAt(delete_ATlist.get(i));
		}

	}





	private static void set_testABC(Instances instances, Instances test_A, Instances test_B, Instances test_C) throws Exception {

		for(int i = 0;i < instances.numInstances();i++){
			/* set A = "vab"
			 * set B = "avb"
			 * set C = "abv"
			 * */

			if(instances.instance(i).toString(7).equals("avb")){
				test_A.instance(i).setMissing(0);
				test_C.instance(i).setMissing(0);
			}else if(instances.instance(i).toString(7).equals("vab")){
				test_B.instance(i).setMissing(0);
				test_C.instance(i).setMissing(0);
			}else{
				test_A.instance(i).setMissing(0);
				test_B.instance(i).setMissing(0);
			}
		}
		test_A.deleteWithMissing(0);
		test_B.deleteWithMissing(0);
		test_C.deleteWithMissing(0);

	}

	private static void set_trainABC(Instances instances, Instances train_A, Instances train_B, Instances train_C) {
		for(int i = 0; i < instances.numInstances();i++){

			/* set A = "vab"
			 * set B = "avb"
			 * set C = "abv"
			 * */

				if(instances.instance(i).toString(7).equals("avb")){
					train_A.instance(i).setMissing(0);
					train_C.instance(i).setMissing(0);
				}else if(instances.instance(i).toString(7).equals("vab")){
					train_B.instance(i).setMissing(0);
					train_C.instance(i).setMissing(0);
				}else if(instances.instance(i).toString(7).equals("abv")){
					train_A.instance(i).setMissing(0);
					train_B.instance(i).setMissing(0);
				}else{    //When a missing value exists
					train_A.instance(i).setMissing(0);
					train_B.instance(i).setMissing(0);
					train_C.instance(i).setMissing(0);
				}
		}

		train_A.deleteWithMissing(0);
		train_B.deleteWithMissing(0);
		train_C.deleteWithMissing(0);
	}

	private static double instancesCoefficient(double kA, double pA, double kB, double pB, double kC, double pC, Instances instances){

		/* kA: coefficient for keyword-related Features in subset A
		 * pA: coefficient for protein-related Features in subset A
		 *
		 * kB: coefficient for keyword-related Features in subset B
		 * pB: coefficient for protein-related Features in subset B
		 *
		 * kC: coefficient for keyword-related Features in subset C
		 * pC: coefficient for protein-related Features in subset C
		 * */

		double F = 0;

		for(int i = 0; i < instances.numInstances();i++){

			/* set A = "vab"
			 * set B = "avb"
			 * set C = "abv"
			 * */

			Instance  instancecheck = instances.instance(i);

				if(instancecheck.toString(7).equals("vab")){/* set A = "vab" */

						instancecheck.setValue(1,  (instancecheck.value(1)) * kA);
						instancecheck.setValue(2,  (instancecheck.value(2)) * kA);
						instancecheck.setValue(3,  (instancecheck.value(3)) * pA);

						instancecheck.setValue(35,  (instancecheck.value(35)) * kA);
						instancecheck.setValue(36,  (instancecheck.value(36)) * kA);
						instancecheck.setValue(37,  (instancecheck.value(37)) * kA);

						instancecheck.setValue(38,  (instancecheck.value(38)) * kA);
						instancecheck.setValue(39,  (instancecheck.value(39)) * kA);
						instancecheck.setValue(40,  (instancecheck.value(40)) * kA);

						instancecheck.setValue(41,  (instancecheck.value(41)) * pA);
						instancecheck.setValue(42,  (instancecheck.value(42)) * pA);
						instancecheck.setValue(43,  (instancecheck.value(43)) * pA);


				}else if(instancecheck.toString(7).equals("avb")){/* set B = "avb" */

						instancecheck.setValue(1,  (instancecheck.value(1)) * kB);
						instancecheck.setValue(2,  (instancecheck.value(2)) * kB);
						instancecheck.setValue(3,  (instancecheck.value(3)) * pB);

						instancecheck.setValue(35,  (instancecheck.value(35)) * kB);
						instancecheck.setValue(36,  (instancecheck.value(36)) * kB);
						instancecheck.setValue(37,  (instancecheck.value(37)) * kB);

						instancecheck.setValue(38,  (instancecheck.value(38)) * kB);
						instancecheck.setValue(39,  (instancecheck.value(39)) * kB);
						instancecheck.setValue(40,  (instancecheck.value(40)) * kB);

						instancecheck.setValue(41,  (instancecheck.value(41)) * pB);
						instancecheck.setValue(42,  (instancecheck.value(42)) * pB);
						instancecheck.setValue(43,  (instancecheck.value(43)) * pB);


				}else if(instancecheck.toString(7).equals("abv")){/* set C = "abv" */

						instancecheck.setValue(1,  (instancecheck.value(1)) * kC);
						instancecheck.setValue(2,  (instancecheck.value(2)) * kC);
						instancecheck.setValue(3,  (instancecheck.value(3)) * pC);

						instancecheck.setValue(35,  (instancecheck.value(35)) * kC);
						instancecheck.setValue(36,  (instancecheck.value(36)) * kC);
						instancecheck.setValue(37,  (instancecheck.value(37)) * kC);

						instancecheck.setValue(38,  (instancecheck.value(38)) * kC);
						instancecheck.setValue(39,  (instancecheck.value(39)) * kC);
						instancecheck.setValue(40,  (instancecheck.value(40)) * kC);

						instancecheck.setValue(41,  (instancecheck.value(41)) * pC);
						instancecheck.setValue(42,  (instancecheck.value(42)) * pC);
						instancecheck.setValue(43,  (instancecheck.value(43)) * pC);

				}
		}



		return F;
	}


}
