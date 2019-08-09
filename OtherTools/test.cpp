#include<iostream>
#include<string>
using namespace std;


void quicksort2(int* ind, double* data, int low, int high)
 {
    if (low >= high) return;

    int i = low, j = high;
    double t = data[low];
    int t2 = ind[low];

    while (i != j) {
        while (i < j && data[j] > t) j--;
        if (i < j) {
            data[i++] = data[j];
            ind[i-1] = ind[j];
        }

        while (i < j && data[i] <= t) i++;
        if (i < j) {
            data[j--] = data[i];
            ind[j+1] = ind[i];
        }
    }

    data[i] = t;
    ind[i] = t2;

    quicksort2(ind, data, low, i-1 );
    quicksort2(ind, data, i+1, high);

    return;
}

int main()
{

    int m = 3;
    double *data = new double[m+2];
    for (int i = 0; i<m; i++)
    {
        data[i] = i + 1;
    }
    data[3] = 1;
    data[4] = 2;
    //
    // data[0] = 1;
    // data[1] = 2;
    // data[2] = 3;
    // data[3] = 1;
    // data[4] = 2;

    int len = 5;
    int num = 4;
    double **result = new double *[num];
    for (int i = 0; i<num; i++)
    {
        result[i] = new double[len];
        memset(result[i],0,len * sizeof(double));
    }

    // for (int i = 0; i < num; i++)
    // {
    //     for (int j = 0; j < len; j++)
    //     result[i][j] = 0;
    // }




    for (int i = 0; i<num; i++)
    {
        result[i] = new double[len];
    }

    // for (int i = 0; i < num; i++)
    // {   cout << "\n";
    //     for (int j = 0; j < len; j++)
    //     {
    //         cout << result[i][j];
    //         cout << "\t";
    //     }
    // }


    double weight = double(len) / double(num);

    double *short_volume = new double[num];
    for (int i = 0; i < num; i++)
    {
        short_volume[i] = weight;
    }

    // for (int i = 0; i < num; i++)
    // {
    //     cout << short_volume[i];
    //     cout << "\n";
    // }

    int *ind = new int[len];
    for (int i = 0; i < len; i++)
        ind[i] = i;

	// for (int i = 0; i < len; i++)
	// {
	// 	for (int j = 0; j < len- i - 1; j++)
	// 	{
	// 		if (data[j] > data[j + 1])
	// 		{
	// 			double temp = data[j];
	// 			data[j] = data[j + 1];
	// 			data[j + 1] = temp;
    //
	// 			int ind_temp = ind[j];
	// 			ind[j] = ind[j + 1];
	// 			ind[j + 1] = ind_temp;
	// 		}
	// 	}
	// }

    quicksort2(ind, data, 0, len-1);

     /*for (int i = 0; i < len; i++)
     {
         cout << data[i];
         cout << "\t";
     }

     cout << "\n";

     for (int i = 0; i < len; i++)
     {
         cout << ind[i];
        cout << "\t";
    }*/

    int ** group = new int *[len];
    for (int i = 0; i < len; i++)
    {
        group[i] = new int[len];
    }

    int group_num = 0;
    int weight_temp = 1;
    int * group_weight = new int[len];

    double pre_value = data[0];
    int pt_0 = 0;
    int pt_1 = 0;

    // cout << "\n";
    group[pt_0][pt_1] = ind[0];
    pt_1 += 1;
    // cout << group[0][0];

    for (int i = 1; i <len; i++)
    {
        if (data[i] == pre_value)
        {
            group[pt_0][pt_1] = ind[i];
            weight_temp += 1;
            pt_1 += 1;
        }

        else
        {
            group_weight[pt_0] = weight_temp;
            weight_temp = 1;
            pt_0 += 1;
            pt_1 = 0;
            group[pt_0][pt_1] = ind[i];
            pt_1 += 1;
            pre_value = data[i];
            group_num += 1;
        }
    }

    if (weight_temp != 0)
    {
        group[pt_0][pt_1] = data[len];
        group_num += 1;
        group_weight[pt_0] = weight_temp;
    }

    // for(int i = 0; i < group_num; i++)
    // {
    //     cout << group_weight[i];
    //     cout << "\n";
    // }

    // for (int i = 0; i< group_num; i++)
    // {
    //     cout << "\n";
    //
    //     for (int j = 0; j < group_weight[i]; j++)
    //     {
    //         cout << group[i][j];
    //         cout <<"\t";
    //     }
    // }

    int pt_ind = 1;
    float left;
    for (int i = 0; i < group_num; i++)
    {
        left = group_weight[i];
        while(left != 0 & pt_ind <= num)
        {
            if (short_volume[pt_ind -1] >= left)
            {
                for (int j = 0; j < group_weight[i]; j++)
                {
                    result[pt_ind-1][group[i][j]] += float(left) / float(group_weight[i]);
                }
                short_volume[pt_ind-1] -= left;
                left = 0;
            }

            else
            {
                for(int j = 0; j<group_weight[i];j++)
                {
                    result[pt_ind - 1][group[i][j]] += float(short_volume[pt_ind-1]) / float(group_weight[i]);
                }
                left -= short_volume[pt_ind - 1];
                short_volume[pt_ind - 1] = 0;
                pt_ind += 1;
            }
        }
    }

    // for (int i = 0; i <num; i++)
    // {
    //     cout << "\n";
    //     for(int j = 0; j< len; j++)
    //     {
    //         cout<< result[i][j];
    //         cout << "\t";
    //     }
    // }

    double *res_new = new double[num * len];
    for (int i = 0; i < num ; i++)
    {
        for (int j = 0; j < len;j++)
        {
            res_new[i*len + j] = result[i][j];
        }
    }

    for (int i = 0; i < num * len;i++)
    {
        cout << res_new[i];
        cout << "\t";
    }



    delete [] short_volume;
    delete [] ind;
    delete [] group_weight;
    for (int i = 0; i <len; i++)
    delete [] group[i];

    for (int i = 0; i < num; i++)
    delete [] result[i];

    return 0;
}
