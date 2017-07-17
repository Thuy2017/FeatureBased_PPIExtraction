
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



/* InstanceLevelCV: for Instance level CV on only one corpus */
public class InstanceLevelCV {

//	public static String corpus = "LLL"; // R=18
	public static String corpus = "HPRD50";// R=40
//	public static String corpus = "IEPA"; // R=44
//  public static String corpus = "AImed";// R=22
//	public static String corpus = "BioInfer";
	public static int R = 40; //for rand number

	public static void main(String[] args) throws Exception{

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


		int date_num = 0;

		int num_folds = 10;



		if(corpus.equals("LLL"))
			date_num = 330;
		else if(corpus.equals("HPRD50"))
			date_num = 433;
		else if(corpus.equals("IEPA"))
			date_num = 817;
		else if(corpus.equals("AImed"))
			date_num = 5834;
		else if(corpus.equals("BioInfer"))
			date_num = 9666;


		DataSource source = new DataSource("ARFFFiles20170613/"+ corpus +".arff");//get data from the file corpus.arf

		int random_num = date_num*R;
		Instances instances = source.getDataSet();  //get instances from source-data

		Instances instances_train;
		Instances instances_test;

		Instances train_A;
		Instances train_B;
		Instances train_C;

		Instances train_Z;


		Instances test_A;
		Instances test_B;
		Instances test_C;

		Instances test_Z;


		double tp = 0,fp = 0,fn = 0;

		double f=0,p=0,r=0;

		double F = 0;
		double P = 0;
		double R = 0;

		int[] rand = new int[random_num];


		make_rand(rand,date_num);   //create random numbers (rand is the array of indexes of instances)

		for(int j = 0;j<random_num;j=j+2)   {
			instances.swap(rand[j], rand[j+1]); //shuffle the instances randomly
		}


		Classifier classifierA = new RandomForest();    //setting classifier
		Classifier classifierB = new RandomForest();
		Classifier classifierC = new RandomForest();


		classifierA.setOptions(weka.core.Utils.splitOptions("-I 10" + " -K 0 " + "  -S 1"));
		classifierB.setOptions(weka.core.Utils.splitOptions("-I 10" + " -K 0 " + "  -S 1"));
		classifierC.setOptions(weka.core.Utils.splitOptions("-I 10" + " -K 0 " + "  -S 1"));

		instances.setClassIndex(instances.numAttributes() - 1);   // Make the last attribute be the class


	/* Check shrink coefficients for each round of the corpus */

		double kA[]={1,1,1,1,1,1,1,1,1,1};
		double pA[]={1,1,1,1,1,1,1,1,1,1};
		double kB[]={1,1,1,1,1,1,1,1,1,1};
		double pB[]={1,1,1,1,1,1,1,1,1,1};
		double kC[]={1,1,1,1,1,1,1,1,1,1};
		double pC[]={1,1,1,1,1,1,1,1,1,1};


		for(int i = 0;i<num_folds;i++){                                  //10-foldsCV

			train_Z = instances.trainCV(num_folds, i);
			test_Z = instances.testCV(num_folds, i);

			train_Z.setClassIndex(train_Z.numAttributes() - 1);  // Make the last attribute be the class
			test_Z.setClassIndex(test_Z.numAttributes() - 1);


			double[] co = new double[6];
			co = CoefficientsofEachRoundCV(9,train_Z, (int)(date_num*0.9), i);

			kA[i] = co[0];
			pA[i] = co[1];
			kB[i] = co[2];
			pB[i] = co[3];
			kC[i] = co[4];
			pC[i] = co[5];




//			kA[i] = 1; pA[i] = 1; kB[i] = 0.9; pB[i] =0.9; kC[i] =0.9; pC[i] =0.9;

			System.out.println("Check shrink coefficients for each round of the corpus: On round = "+ i +" of CV: With " + " kA[i] = " +kA[i]  + "; pA[i] = " +pA[i]  + "; kB[i] = " +kB[i] + "; pB[i] = " +pB[i] + "; kC[i] = " +kC[i]  + "; pC[i] = " +pC[i] +";" ) ;
//			System.out.println("instances.numInstances() = " + instances.numInstances());
//			System.out.println("train_Z.numInstances() = " + train_Z.numInstances());
//			System.out.println("test_Z.numInstances() = " + test_Z.numInstances());
		}



		//i=0
 /*		LLL R=40
  * 	kA[0] = 1.0; pA[0] = 1.0; kB[0] = 1.0; pB[0] = 1.0; kC[0] = 1.0; pC[0] = 1.0;
 		// i=1
		kA[1] = 0.9; pA[1] = 0.9; kB[1] = 0.9; pB[1] = 0.9; kC[1] = 1.0; pC[1] = 0.9;
		// i=2;
		kA[2] = 1.0; pA[2] = 0.9; kB[2] = 1.0; pB[2] = 0.9; kC[2] = 0.9; pC[2] = 1.0;
		// i=3;
		kA[3] = 0.9; pA[3] = 1.0; kB[3] = 1.0; pB[3] = 1.0; kC[3] = 0.9; pC[3] = 0.9;
		// i=4;
		kA[4] = 1.0; pA[4] = 1.0; kB[4] = 1.0; pB[4] = 0.9; kC[4] = 0.9; pC[4] = 0.9;
		// i=5;
		kA[5] = 0.9; pA[5] = 0.9; kB[5] = 0.9; pB[5] = 1.0; kC[5] = 1.0; pC[5] = 1.0;
		// i=6;
		kA[6] = 1.0; pA[6] = 0.9; kB[6] = 0.9; pB[6] = 1.0; kC[6] = 0.9; pC[6] = 1.0;
		// i=7;
		kA[7] = 0.9; pA[7] = 0.9; kB[7] = 0.9; pB[7] = 0.9; kC[7] = 0.9; pC[7] = 1.0;
		// i=8;
		kA[8] = 0.9; pA[8] = 1.0; kB[8] = 1.0; pB[8] = 1.0; kC[8] = 0.9; pC[8] = 1.0;

		// i=9;
		kA[9] = 0.9; pA[9] = 1.0; kB[9] = 0.9; pB[9] = 0.9; kC[9] = 1.0; pC[9] = 1.0;
*/
	/* End: Check shrink coefficients for each round of the corpus */


		//instances = source.getDataSet();
		for(int i = 0;i<num_folds;i++){                                  //10-foldsCV
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


			/* Shrinking coefficients */
//			instancesCoefficient(kA[i], pA[i], kB[i], pB[i], kC[i], pC[i], instances_train);
//			instancesCoefficient(kA[i], pA[i], kB[i], pB[i], kC[i], pC[i], instances_test);

			instancesCoefficient(kA[i], pA[i], kB[i], pB[i], kC[i], pC[i], train_A);
			instancesCoefficient(kA[i], pA[i], kB[i], pB[i], kC[i], pC[i], train_B);
			instancesCoefficient(kA[i], pA[i], kB[i], pB[i], kC[i], pC[i], train_C);


			instancesCoefficient(kA[i], pA[i], kB[i], pB[i], kC[i], pC[i], test_A);
			instancesCoefficient(kA[i], pA[i], kB[i], pB[i], kC[i], pC[i], test_B);
			instancesCoefficient(kA[i], pA[i], kB[i], pB[i], kC[i], pC[i], test_C);
			/* End: Shrinking coefficients */


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

		System.out.println();

		System.out.println("CROSS-VALIDATION RESULTS of 3subCV-FSShrink on " + corpus + ":" );

	//	System.out.println("InstanceLevelCV on only ONE CORPUS: With " + " kA = " +kA  + "; pA = " +pA  + "; kB = " +kB  + "; pB = " +pB + "; kC = " +kC  + "; pC = " +pC ) ;
		System.out.print("P = "+P/10);
		System.out.print("; R = "+R/10);
		System.out.println("; F = "+F/10);


	}// end main


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


		System.out.println("InstanceLevelCV on only ONE CORPUS: corpus " +corpus + " : On round = "+ round +" of CV: With " + " kA_max = " +kA_max  + "; pA_max = " +pA_max  + "; kB_max = " +kB_max  + "; pB_max = " +pB_max + "; kC_max = " +kC_max  + "; pC_max = " +pC_max +";" ) ;
		System.out.println("F_max = "+F_max);

		double[] r = new double[6];
		r[0]= kA_max;
		r[1]= pA_max;
		r[2]= kB_max;
		r[3]= pB_max;
		r[4]= kC_max;
		r[5]= pC_max;

		return r;

	}

	private static void make_rand(int[] rand,int date_num){          //create random numbers (rand is the array of indexes of instances)
		int num = date_num*R;
		Random random = new Random(1);
		for(int i = 0;i<num;i++){
			rand[i] = Math.abs(random.nextInt())%date_num;
		}
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
		//Instances instances = source.getDataSet();

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


		make_rand(rand,date_num);  //create random numbers (rand is the array of indexes of instances)

		for(int j = 0;j<random_num;j=j+2)   {
			instances.swap(rand[j], rand[j+1]);// shuffle the instances randomly
		}


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


