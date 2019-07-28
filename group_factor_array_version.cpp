#include<iostream>
#include<string>
using namespace std;

void quicksort2(int* ind, double* data, int low, int high)
 {
    if (low >= high) return;

    int i = low, j = high;
    double t = data[low];
    int t2 = ind[low];

    while (i != j)
    {
        while (i < j && data[j] > t) j--;
        if (i < j)
        {
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


int search_left(int ind_now, double* data)
{
    double value_now = data[ind_now];
    while(ind_now > 0)
    {
        ind_now -= 1;
        if(data[ind_now] == value_now) continue;
        else return ind_now+1;
    }

    return ind_now;
}


int search_right(int ind_now, double* data,int length)
{
    double value_now = data[ind_now];
    while(ind_now < length -1)
    {
        ind_now += 1;
        if(data[ind_now] == value_now) continue;
        else return ind_now;
    }

    return ind_now +1;
}

double * _quantize_factor(double* factor_data, int length, int num)
{
    // 初始化最后的返回指针
    double *result = new double [num * length];
    memset(result,0,(num * length) *sizeof(double));

//  初始化short_volume
    double weight = double(length) / double(num);
    double *short_volume = new double[num];
    for (int i = 0; i < num; i++)
    {
        short_volume[i] = weight;
    }


// 建立一个新的索引array然后对factor_data和索引序列进行排序
    int *ind = new int[length];
    for (int i = 0; i < length; i++)
    ind[i] = i;

    quicksort2(ind, factor_data, 0, length-1);


// 初始化分组的内存 长度 * 长度
    int ** group = new int *[length];
    for (int i = 0; i < length; i++)
    {
        group[i] = new int[length];
    }

    int group_num = 0;
    int weight_temp;
    int* group_weight = new int[2*num - 1];

    int* support_points = new int[2*num];
    support_points[0] = 0;
    support_points[2*num - 1] = length;
    int temp;
    for(int i = 1; i < num; i++)
    {   temp = double(length) / double(num) * i;
        support_points[2*i-1] = search_left(temp, factor_data);
        support_points[2*i] = search_right(temp, factor_data, length);
    }

    int pre_max = 0;
    int pt_1;
    int j = 0;
    while(j < 2*num - 1)
    {
        if((support_points[j] >= support_points[j+1])||(support_points[j] < pre_max))
        {
            j += 1;
        }

        else
        {
            weight_temp = support_points[j + 1] - support_points[j];
            // cout << support_points[j]<< endl;
            // cout << support_points[j+1] << endl;
            group_weight[group_num] = weight_temp;
            // cout << group_weight[0] << endl;
            pt_1 = 0;
            for(int k = support_points[j]; k < support_points[j+1];k++)
            {
                group[group_num][pt_1] = ind[k];
                pt_1 += 1;
            }
            pre_max = support_points[j+1];
            group_num += 1;
            j += 1;
        }
    }


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
                    result[(pt_ind-1) * length + group[i][j]] += float(left) / float(group_weight[i]);
                }
                short_volume[pt_ind-1] -= left;
                left = 0;
            }

            else
            {
                for(int j = 0; j<group_weight[i];j++)
                {
                    result[(pt_ind-1) * length + group[i][j]] += float(short_volume[pt_ind-1]) / float(group_weight[i]);
                }
                left -= short_volume[pt_ind - 1];
                short_volume[pt_ind - 1] = 0;
                pt_ind += 1;
            }
        }
    }



    delete [] short_volume;
    delete [] ind;
    delete [] group_weight;

    for (int i = 0; i <length; i++)
    delete [] group[i];

    return result;

}


int main()
{   int m = 3;
    double *data = new double[m+2];
    for (int i = 0; i<m; i++)
    {
        data[i] = i + 1;
    }
    data[3] = 1;
    data[4] = 2;

    // for (int i = 0; i < m+2; i++)
    // data[i] = 1;

    double *result;


    result = _quantize_factor(data, 5, 4);

    for (int i = 0; i < 20; i++)
    {
        cout << result[i];
        cout << "\t";
    }

    return 0;
}
