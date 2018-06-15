/***************************************************************************
 * 	
 * 	But: Fichier servant à récupérer et définir les paramètres du calcul 
 *		 réalisé par la fonction "svd_with_t" définie dans "svd_with_t.cpp".
 * 
 *	Usage: ./main <N> <Re> <alpha> <tMax> <dt> <step>
 * 		
 * 		N: nombre de points sur un côté de la grille
 *		Re: nombre de Reynolds
 *  	alpha: nombre d'onde des modes propres
 *		tMax: temps maximal jusqu'auquel calculer la svd
 *		dt: pas de temps
 *		step: svd réalisée toute les <step> étapes, i.e. intervalle de
 *			  <dt>*<step> entre chaque instant auquel la svd est réalisée
 * 
 * 	Les résultats du calcul sont écrits dans le fichier ./data/data
 * 
 * 	Auteur: Cédric Schoonen
 ***************************************************************************/ 

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "magma.h"
#include "svd_with_t.h" // code qui contient l'implémentation du calcul de svd

int main(int argc, char ** argv)
{
	/* Déclaration des paramètres par défaut */
	
	int N = 401;
	double re = 10000;
	double alpha = 0.6;
	double tMax = 100;
	double dt = 0.1;
	int step = 10;
	
	/* Récupération des paramètres donnés en argument */
	
	if (argc>1){N = atoi(argv[1]);} 	// attention mauvais inputs
	if (argc>2){re = atof(argv[2]);} 	// attention mauvais inputs
	if (argc>3){alpha = atof(argv[3]);} // attention mauvais inputs
	if (argc>4){tMax = atof(argv[4]);} 	// attention mauvais inputs
	if (argc>5){dt = atof(argv[5]);} 	// attention mauvais inputs
	if (argc>6){step = atoi(argv[6]);} 	// attention mauvais inputs
	
	/* Lancement du calcul des valeurs singulières */
	
	magma_init();
	
	svd_with_t(re,alpha,N,tMax,dt,step);	// réalise la svd avec les
											// paramètres voulus
	
	magma_finalize();
	
	return 0;
}