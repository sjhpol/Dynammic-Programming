// This script is just to generate the executable that we need for testing calling it inside Python.

#include <stdio.h>
#include <time.h>

int main() {
  FILE *fp;
  time_t current_time;
  char time_string[100];

  // Open the file for writing in text mode
  fp = fopen("current_time.txt", "w");

  // Check if file opened successfully
  if (fp == NULL) {
    printf("Error opening file!\n");
    return 1;
  }

  // Get current time
  current_time = time(NULL);

  // Convert time to a string format
  strftime(time_string, sizeof(time_string), "%c", localtime(&current_time));

  // Write the time string to the file
  fprintf(fp, "%s\n", time_string);

  // Close the file
  fclose(fp);

  printf("Current time written to 'current_time.txt'\n");

  return 0;
}