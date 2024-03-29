/**
 * Copyright 2019 DayHR Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

//
// Created by lcy on 2019-07-26.
//

#ifndef OCR_INCLUDE_LOG_HPP
#define OCR_INCLUDE_LOG_HPP

#define MSG_DEBUG  0x01
#define MSG_INFO   0x02
#define MSG_WARING 0x03
#define MSG_ERROR  0x04

#ifndef PRINT_LEVEL
#define PRINT_LEVEL
static int print_level = MSG_DEBUG | MSG_INFO | MSG_WARING | MSG_ERROR;
#endif  // PRINT_LEVEL

/**
 * print error.
 * example: "[MSG_ERROR][parser_URL(101)]:url invaild"
 * @ author: Changyu Liu
 * @ last modifly time: 2019.7.26
 */
#define lprintf(level, format, argv...)                                       \
  do {                                                                        \
    if (level & print_level)                                                  \
      printf("[%s][%s(%d)]:" format, #level, __FUNCTION__, __LINE__, ##argv); \
  } while (0)


#endif // OCR_INCLUDE_LOG_HPP
