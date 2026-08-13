#include <iostream>
using namespace std;

int main()
{
    int arr[6] = {2, 4, 6, 8, 9, 15};
    int target;

    cout << "Enter the target sum: ";
    cin >> target;

    int first = 0;
    int second = 0;
    int sum = 0;
    bool found = false;

    while (second < 6)
    {
        sum = sum + arr[second];

        while (sum > target && first <= second)
        {
            sum = sum - arr[first];
            first++;
        }

        if (sum == target)
        {
            cout << "Subarray found from index "
                 << first << " to " << second << endl;

            cout << "Elements are: ";

            for (int i = first; i <= second; i++)
            {
                cout << arr[i] << " ";
            }

            cout << endl;

            found = true;
            break;
        }

        second++;
    }

    if (!found)
    {
        cout << "No continuous subarray found." << endl;
    }

    return 0;
}