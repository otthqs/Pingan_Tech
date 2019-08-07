#include <iostream>
#include <fstream>
#include <vector>
using namespace std;


vector<vector<float>> _quantize_factor(vector<float> factor_data, int num)
{
    vector<vector<float>> result(num,vector<float>(factor_data.size(),0));
    float weight = factor_data.size() / float(num);
    vector<float> short_volume(num,weight);

    vector<vector<float>> data;
    vector<float> temp;

    for(int i = 0;i<factor_data.size();i++)
    {

        temp.push_back(factor_data[i]);
        temp.push_back(i);
        data.push_back(temp);
        temp.clear();

    }

    sort(data.begin(),data.end());

    float pre_value = data[0][0];

    vector<vector<float>> group;
    vector<float> sub;

    for(int i = 0; i<data.size();i++)
    {
        if(data[i][0] == pre_value)
        {
            sub.push_back(data[i][1]);
        }
        else
        {
            group.push_back(sub);
            pre_value = data[i][0];
            sub.clear();
            sub.push_back(data[i][1]);
        }


    }

    if (not sub.empty())
    {
        group.push_back(sub);
    }


    int ind = 1;
    float left;
    for(int i = 0; i<group.size();i++)
    {
        left = group[i].size();
        while(left != 0 & ind <= num)
        {
            if (short_volume[ind - 1] >= left)
            {
                for(int j = 0; j<group[i].size();j++)
                {
                    result[ind - 1][group[i][j]] += float(left) / float(group[i].size());
                }
                short_volume[ind-1] -= left;
                left = 0;
            }

            else if(short_volume[ind-1] < left)
            {
                for(int j = 0; j<group[i].size();j++)
                {
                    result[ind - 1][group[i][j]] += float(short_volume[ind-1]) / float(group[i].size());
                }
                left -= short_volume[ind - 1];
                short_volume[ind - 1] = 0;
                ind += 1;



            }

        }

    }

    return result;

}


int main()
{
    int num = 4;
    vector<float> factor_data;
    for(int i = 1; i < 4; i++)
    {
        factor_data.push_back(i);

    }

    factor_data.push_back(1);
    factor_data.push_back(2);

    // factor_data = [1,2,3,1,2]

    vector<vector<float>> data;
    vector<float> temp;
    for(int i = 0;i<factor_data.size();i++)
    {

        temp.push_back(factor_data[i]);
        temp.push_back(i);
        data.push_back(temp);
        temp.clear();
    }

    sort(data.begin(),data.end());

    // cout << data[1].size();

// data = [[1,0],[1,3],[2,1],[2,4],[3,2]]

    float weight = data.size() / float(num);

    vector<float> short_volume(num,weight);

    vector<vector<float>> result(num,vector<float>(factor_data.size(),0));

    float pre_value = data[0][0];

    vector<vector<float>> group;
    vector<float> sub;


    for(int i = 0; i<data.size();i++)
    {
        if(data[i][0] == pre_value)
        {
            sub.push_back(data[i][1]);
        }
        else
        {
            group.push_back(sub);
            pre_value = data[i][0];
            sub.clear();
            sub.push_back(data[i][1]);
        }


    }

    if (not sub.empty())
    {
        group.push_back(sub);
    }

    // for(int i = 0;i<group.size();i++)
    // {
    //     cout << "\n";
    //     for (int j = 0; j<group[i].size();j++)
    //     {
    //          cout << group[i][j];
    //     }
    // }

    int ind = 1;
    float left;
    for(int i = 0; i<group.size();i++)
    {
        left = group[i].size();
        while(left != 0 & ind <= num)
        {
            if (short_volume[ind - 1] >= left)
            {
                for(int j = 0; j<group[i].size();j++)
                {
                    result[ind - 1][group[i][j]] += float(left) / float(group[i].size());
                }
                short_volume[ind-1] -= left;
                left = 0;
            }

            else if(short_volume[ind-1] < left)
            {
                for(int j = 0; j<group[i].size();j++)
                {
                    result[ind - 1][group[i][j]] += float(short_volume[ind-1]) / float(group[i].size());
                }
                left -= short_volume[ind - 1];
                short_volume[ind - 1] = 0;
                ind += 1;



            }

        }

    }

    for(int i = 0;i<result.size();i++)
    {
        cout << "\n";
        for (int j = 0; j<result[i].size();j++)
        {
            cout << "\t";
             cout << result[i][j];
        }
    }

    return 0;
}
